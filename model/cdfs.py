"""
encoder resnet50
"""
from re import S
import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import *
from .detection_head import *
import torchvision.models.detection as detection


class FewShotSeg(nn.Module):

    def __init__(self, args):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Encoder
        self.encoder = Res50Encoder(replace_stride_with_dilation=[True, True, False],
                                    pretrained_weights="COCO")  # or "ImageNet"

        self.scaler = 20.0
        self.args = args
        self.reference_layer1 = nn.Linear(512, 2, bias=True)
        self.epsilon_list = [0.01, 0.03, 0.05, 0.001, 0.003, 0.005]

        self.Detection_head = Network()

        for param in self.Detection_head.parameters():
            param.requires_grad = False

        self.criterion = nn.NLLLoss(ignore_index=255, weight=torch.FloatTensor([0.1, 1.0]).cuda())
        self.margin = 0.

        self.channel = nn.Conv2d(in_channels=507, out_channels=512, kernel_size=1, stride=1)

        self.mse_loss = nn.MSELoss()

        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, opt, train=False):

        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors  (1, 3, 257, 257)
            qry_mask: label
                N x 2 x H x W, tensor
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size) 

        ## Feature Extracting With ResNet Backbone
        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)

        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view( 
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])
        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(  
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])

        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]  
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        outputs_qry = []
        outputs_qry_coarse = []
        for epi in range(supp_bs):

            """
            supp_fts[[epi], way, shot]: (B, C, H, W) 
            """

            if supp_mask[[0], 0, 0].max() > 0.:

                spt_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                             for shot in range(self.n_shots)] for way in range(self.n_ways)]
                spt_fg_proto = self.getPrototype(spt_fts_)

                supp_fts_b = [[self.getFeatures(supp_fts[[epi], way, shot], 1. - supp_mask[[epi], way, shot])
                               for shot in range(self.n_shots)] for way in range(self.n_ways)]
                spt_bg_proto = self.getPrototype(supp_fts_b)

                # obtain coarse mask of query *******************
                qry_pred = torch.stack(
                    [self.getPred(qry_fts[way], spt_fg_proto[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  
                
                # Combine predictions of different feature maps #
                qry_pred_coarse = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)

                # the loss for coarse prediction
                if train:
                    preds_coarse = torch.cat((1.0 - qry_pred_coarse, qry_pred_coarse), dim=1)
                    outputs_qry_coarse.append(preds_coarse)

                if qry_pred_coarse[epi].max() > 0.:

                    proto_emd = [[self.EMD(supp_fts[way][shot], qry_fts[way], supp_imgs[way][shot], qry_imgs[way],
                                       supp_mask[[epi], way, shot], qry_pred_coarse[epi])
                              for shot in range(self.n_shots)] for way in range(self.n_ways)]
                else:
                    proto_emd = [spt_fg_proto]
                
        


                # structure aware transform
                supp_fts = [[self.structure(supp_fts[way][shot], supp_imgs[way][shot], supp_mask[[epi], way, shot])
                             for shot in range(self.n_shots)] for way in range(self.n_ways)]

                qry_fts = [self.structure(qry_fts[way], qry_imgs[way], qry_pred_coarse) for way in range(self.n_ways)]

                spt_fg_fts = [[self.get_fg(supp_fts[way][shot], supp_mask[[0], way, shot])
                               for shot in range(self.n_shots)] for way in range(self.n_ways)]  

                qry_fg_fts = [self.get_fg(qry_fts[way], qry_pred_coarse[epi])
                              for way in range(self.n_ways)]  

                spt_proto = [self.get_proto_new(spt_fg_fts[epi][way]) for way in range(self.n_ways)]
                qry_proto = [self.get_proto_new(qry_fg_fts[way]) for way in range(self.n_ways)]

                fg_proto = [spt_proto[way] + qry_proto[way] +  proto_emd[epi][way]
                            for way in range(self.n_ways)]

                pred = torch.stack(
                    [self.getPred(qry_fts[way], fg_proto[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  
                pred_up = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)
                pred = torch.cat((1.0 - pred_up, pred_up), dim=1)
                outputs_qry.append(pred)

            else:
                ########################acquiesce prototypical network ################
                supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                              for shot in range(self.n_shots)] for way in range(self.n_ways)]
                fg_prototypes = self.getPrototype(supp_fts_) 

                qry_pred = torch.stack(
                    [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  
                ########################################################################

                # Combine predictions of different feature maps #
                qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
                preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)

                outputs_qry.append(preds)

        output_qry = torch.stack(outputs_qry, dim=1)
        output_qry = output_qry.view(-1, *output_qry.shape[2:])

        output_qry_coarse = torch.stack(outputs_qry_coarse, dim=1)
        output_qry_coarse = output_qry_coarse.view(-1, *output_qry_coarse.shape[2:])

        return output_qry, output_qry_coarse

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5) 

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  

        return fg_prototypes

    def get_fg(self, fts, mask):

        """
        :param fts: (1, C, H', W')
        :param mask: (1, H, W)
        :return: (1, C, N)  N: the number of foreground pixels
        """

        mask = torch.round(mask)
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        mask = mask.unsqueeze(1).bool()
        result_list = []

        for batch_id in range(fts.shape[0]):
            tmp_tensor = fts[batch_id]  
            tmp_mask = mask[batch_id]  

            foreground_features = tmp_tensor[:, tmp_mask[0]]  

            if foreground_features.shape[1] == 1:  
              
                foreground_features = torch.cat((foreground_features, foreground_features), dim=1)

            result_list.append(foreground_features)  

        foreground_features = torch.stack(result_list)

        return foreground_features

    def get_proto_new(self, fts):
        """

        :param fts:  (1, 512, N)
        :return:
        """
        N = fts.size(2)
        proto = torch.sum(fts, dim=2) / (N + 1e-5)

        return proto

    def Transformation_Feature(self, feature, prototype_f, prototype_b):

        """
        supp_fts: support feature (B, C, H, W)
        qry_fts: query feature (B, C, H, W)
        prototype_f: foreground prototype
        prototype_b: background prototype
        """

        bsz = feature.shape[0]
        C = torch.cat((prototype_b.unsqueeze(1), prototype_f.unsqueeze(1)), dim=1)
        eps = 1e-5
        R = self.reference_layer1.weight.expand(C.shape)
        power_R = ((R * R).sum(dim=2, keepdim=True)).sqrt()
        R = R / (power_R + eps)
        power_C = ((C * C).sum(dim=2, keepdim=True)).sqrt()
        C = C / (power_C + eps)

        P = torch.matmul(torch.pinverse(C), R)
        P = P.permute(0, 2, 1)
        init_size = feature.shape
        feature = feature.view(bsz, C.size(2), -1)
        transformed_fts = torch.matmul(P, feature).view(init_size)

        return transformed_fts

    def compute_gradients(self, tensor):
        # Sobel
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

        sobel_x = sobel_x.repeat(tensor.size(1), 1, 1, 1) 
        sobel_y = sobel_y.repeat(tensor.size(1), 1, 1, 1) 

        if tensor.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()

        edge_x = F.conv2d(tensor, sobel_x, padding=1, groups=tensor.size(1))
        edge_y = F.conv2d(tensor, sobel_y, padding=1, groups=tensor.size(1))

        gradient_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        return gradient_magnitude

    def local_variance(self, images, kernel_size=3):
        """
        计算图像的局部方差。
        :param images: 输入的图像，尺寸为 (N, C, H, W)
        :param kernel_size: 局部方差计算的窗口大小
        :return: 每张图像的局部方差
        """
        N, C, H, W = images.size()

        mean = F.avg_pool2d(images, kernel_size, stride=1, padding=kernel_size // 2)

        mean_of_square = F.avg_pool2d(images ** 2, kernel_size, stride=1, padding=kernel_size // 2)

        local_var = mean_of_square - mean ** 2

        var_per_image = local_var.view(N, C, -1).mean(dim=2)

        return var_per_image

    def structure(self, spt_fts, supp_imgs, spt_mask):

        """

        :param spt_fts: (B, C, H', W')
        :param qry_fts: (B, C, H', W')
        :param supp_imgs: (B, C, H, W)
        :param qry_imgs: (B, C, H, W)
        :param spt_mask: (B, H, W)
        :return:
        """

        spt_fts = F.interpolate(spt_fts, size=supp_imgs.shape[-2:], mode='bilinear')

        spt_fg = supp_imgs * spt_mask
        boundary_mask = self.Detection_head(spt_fg)

        spt_fts_boundary_aware = boundary_mask * spt_fts  


        gradinet_map = self.compute_gradients(spt_fts) 
        gradient_map_fg = gradinet_map * spt_mask 
        local_variance_map = self.local_variance(gradient_map_fg)

        _, top_indices = torch.topk(local_variance_map, k=5)

        channels_to_keep = torch.ones(spt_fts.size(1), dtype=bool)
        channels_to_keep[top_indices] = False
        spt_fts_remaining = spt_fts[:, channels_to_keep, :, :]
        spt_fts_remaining = self.channel(spt_fts_remaining)

        return spt_fts_remaining + spt_fts_boundary_aware

    def structural_weight_map(self, fts, img, mask):

        fts = F.interpolate(fts, size=img.shape[-2:], mode='bilinear')
        # calculate gradient
        gradinet_map = self.compute_gradients(fts) 
        gradient_map_fg = gradinet_map * mask 
        local_variance_map = self.local_variance(gradient_map_fg)  
        
        # normalization
        weight = local_variance_map / local_variance_map.sum()
        # relu activation
        weight = F.relu(weight) + 1e-5
        return weight

    def EMD(self, spt_fts, qry_fts, spt_img, qry_img, spt_mask, qry_mask):

        """
        :param spt_fts: (1, C, h, w)
        :param qry_fts: (1, C, h, w)
        :param spt_img: (1, 3, H, W)
        :param qry_img: (1, 3, H, W)
        :param spt_mask: (1, H, W)
        :param qry_mask: (1, H, W)
        :return:
        """

        spt_fts_fg = self.get_fg(spt_fts, spt_mask)  
        qry_fts_fg = self.get_fg(qry_fts, qry_mask)  

        foreground_size = 128
        num_stacks = 4
        stack_size = foreground_size / num_stacks  
        num_dimension = spt_fts_fg.shape[1]  

        num_node = num_dimension

        pool_opt = nn.AdaptiveAvgPool2d((num_dimension, foreground_size))

        spt_fts_fg = pool_opt(spt_fts_fg.unsqueeze(0))

        qry_fts_fg = pool_opt(qry_fts_fg.unsqueeze(0))

        spt_fts_fg, qry_fts_fg = spt_fts_fg.squeeze(0), qry_fts_fg.squeeze(
            0)  

        weight_spt = self.structural_weight_map(spt_fts, spt_img, spt_mask) 
        weight_qry = self.structural_weight_map(qry_fts, qry_img, qry_mask)  

        # split to stacks
        spt_fts_fg_split = torch.chunk(spt_fts_fg, num_stacks, -1)  
        qry_fts_fg_split = torch.chunk(qry_fts_fg, num_stacks, -1)  

        score_list = []
        for i, data in enumerate(spt_fts_fg_split):
            spt_fts_fg_stack, qry_fts_fg_stack = spt_fts_fg_split[i], qry_fts_fg_split[i]  
            spt_fts_fg_stack, qry_fts_fg_stack = spt_fts_fg_stack.permute(0, 2, 1), qry_fts_fg_stack.permute(0, 2, 1)
            # (1, stack_size, num_node)

            similarity_map = self.get_similiarity_map(spt_fts_fg_stack, qry_fts_fg_stack)  
            # employ opencv's EMD
            _, flow = self.emd_inference_opencv(1 - similarity_map[0, 0, :, :], weight_spt[0],
                                                weight_qry[0])  
            similarity_map[0, 0, :, :] = (similarity_map[0, 0, :, :]) * torch.from_numpy(flow).cuda()
            temperature = (12.5 / num_dimension)
            score = similarity_map.sum(-1).sum(-1) * temperature
            score_list.append(score)

        score = torch.cat(score_list, dim=-1)
        # activation, distance map transfers into weighting map
        score_max = torch.max(score)
        score_min = torch.min(score)
        score_normalized = (score - score_min) / (score_max - score_min)

        weights = torch.exp(- score_normalized)  

        resorted_spt_fts_list = [spt_fts_fg_split[i] * weights[0, i] for i in range(num_stacks)]

        resorted_qry_fts_list = [qry_fts_fg_split[i] * weights[0, i] for i in range(num_stacks)]

        elements = []
        for i in range(num_stacks):
            resorted_spt_element, resorted_qry_element = resorted_spt_fts_list[i], resorted_qry_fts_list[
                i]  
            resorted_spt_element, resorted_qry_element = resorted_spt_element.squeeze(0), resorted_qry_element.squeeze(
                0)  
            element_a = (self.mlp(resorted_spt_element) + self.mlp(resorted_qry_element)) * weights[0, i]
            element_b = self.mlp((resorted_spt_element + resorted_qry_element) * weights[0, i])
            element = self.mlp(element_a + element_b)
            elements.append(element)
        element = torch.cat(elements, dim=-1)
        element = element.unsqueeze(0)  
        L = element.size(2)
        proto = torch.sum(element, dim=2) / (L + 1e-5)

        return proto

    def get_similiarity_map(self, spt_fts_fg, qry_fts_fg):

        """
        cosine similiarity
        spt_fts_fg: foreground support feature  --->  (1, foreground_size, num_node)
        qry_fts_fg: foreground query feature  --->  (1, foreground_size, num_node)
        """

        num_node = spt_fts_fg.shape[-1]
        spt_fts_fg, qry_fts_fg = spt_fts_fg.unsqueeze(0), qry_fts_fg.unsqueeze(0)  

        spt_fts_fg = spt_fts_fg.permute(0, 1, 3, 2)  
        qry_fts_fg = qry_fts_fg.permute(0, 1, 3, 2)  

        feature_size = spt_fts_fg.shape[-2]  
        spt_fts_fg = spt_fts_fg.unsqueeze(-3)  
        qry_fts_fg = qry_fts_fg.unsqueeze(-2)  
        qry_fts_fg = qry_fts_fg.repeat(1, 1, 1, feature_size, 1) 
        similarity_map = F.cosine_similarity(spt_fts_fg, qry_fts_fg,
                                             dim=-1)  

        return similarity_map

    def emd_inference_opencv(self, cost_matrix, weight1, weight2):

        """
        :param cost_matrix: (n_elements, n_elements)
        :param weight1: (n_elements)
        :param weight2: (n_elements)
        :return:
        """
        cost_matrix = cost_matrix.detach().cpu().numpy()

        weight1 = F.relu(weight1) + 1e-5
        weight2 = F.relu(weight2) + 1e-5

        weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
        weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

        cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)

        return cost, flow

