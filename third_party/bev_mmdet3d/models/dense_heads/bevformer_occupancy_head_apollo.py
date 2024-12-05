# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

import copy
from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
import math

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from .bevformer_occupancy_head import BEVFormerOccupancyHead
from mmdet.core.bbox import build_bbox_coder
from ...core.bbox.util import normalize_bbox, denormalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding,build_transformer_layer_sequence
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.builder import build_loss, build_head
from mmcv.ops import points_in_boxes_part
from ..utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from ..utils.visual import save_tensor
# from projects.mmdet3d_plugin.models.occ_loss_utils import lovasz_softmax, CustomFocalLoss
# from projects.mmdet3d_plugin.models.occ_loss_utils import nusc_class_frequencies, nusc_class_names
# from projects.mmdet3d_plugin.models.occ_loss_utils import geo_scal_loss, sem_scal_loss, CE_ssc_loss

from mmcv.runner import get_dist_info

@HEADS.register_module()
class BEVFormerOccupancyHeadApollo(BEVFormerOccupancyHead):
    def __init__(self,
                 *args,
                 group_detr=1,
                 occ_tsa=None,
                 positional_encoding_occ=None,
                 balance_cls_weight=False,
                 loss_lovasz=None,
                 loss_affinity=None,
                 **kwargs):
        self.group_detr = group_detr
        assert 'num_query' in kwargs
        kwargs['num_query'] = group_detr * kwargs['num_query']
        loss_type = {
            'FocalLoss': 'focal_loss',
            'CustomFocalLoss': 'CustomFocalLoss',
            'CrossEntropyLoss': 'ce_loss'
        }
        kwargs['occ_loss_type'] = loss_type[kwargs['loss_occupancy']['type']]
        super().__init__(*args, **kwargs)
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, self.occ_zdim*self.occ_dims, kernel_size=1),
            nn.BatchNorm2d(self.occ_zdim*self.occ_dims),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.occ_zdim*self.occ_dims, self.occ_zdim*self.occ_dims, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.occ_zdim*self.occ_dims),
            nn.ReLU(inplace=True),
        )
        if occ_tsa:
            self.upsample_layer = nn.Sequential(
                    nn.ConvTranspose2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(self.embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
                    nn.BatchNorm2d(self.embed_dims),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(self.embed_dims),
                    nn.ReLU(inplace=True),
                )
            self.occ_tsa = build_transformer_layer_sequence(occ_tsa)
            self.occ_tsa_head = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.occ_zdim*self.occ_dims, kernel_size=1),
                nn.BatchNorm2d(self.occ_zdim*self.occ_dims),
                nn.ReLU(inplace=True)
            )
            if positional_encoding_occ is not None:
                    positional_encoding_occ['row_num_embed'] = self.occ_xdim
                    positional_encoding_occ['col_num_embed'] = self.occ_ydim

                    self.positional_encoding_occ = build_positional_encoding(
                        positional_encoding_occ)
                    assert 'num_feats' in positional_encoding_occ
                    num_feats = positional_encoding_occ['num_feats']
                    assert num_feats * 2 == self.embed_dims, 'embed_dims should' \

                    f' be exactly 2 times of num_feats. Found {self.embed_dims}' \

                    f' and {num_feats}.'
            else:
                self.positional_encoding_occ = None
        else:
            self.occ_tsa = None

    def upsample_tsa_occ(self, feat_flatten, spatial_shapes, level_start_index, bev_for_occ, bs, seq_len, **kwargs):
        bev_for_occ = bev_for_occ.permute(1, 2, 0).contiguous().view(bs*seq_len, -1, self.bev_h, self.bev_w) # (bev_h*bev_w, bs*len_seq, c) -> (bs*len_seq, c, bev_h, bev_w)
        upsampled_bev_embed = self.upsample_layer(bev_for_occ) # (bs*len_seq, c, occ_xdim, occ_ydim)
        bev_queries = upsampled_bev_embed.flatten(2).permute(2, 0, 1) # (occ_xdim*occ_ydim, bs*len_seq, c)
        dtype = feat_flatten.dtype
        occ_bev_mask = torch.zeros((bs, self.occ_xdim, self.occ_ydim),
                                    device=bev_queries.device).to(dtype)
        query_pos = self.positional_encoding_occ(occ_bev_mask).to(dtype)
        query_pos = query_pos.flatten(2).permute(0, 2, 1)
        bev_embed = self.occ_tsa(
            bev_queries,
            feat_flatten,
            feat_flatten,
            query_pos=query_pos,
            bev_h=self.occ_xdim,
            bev_w=self.occ_ydim,
            bev_pos=torch.zeros_like(bev_queries).permute(1, 0, 2),  # fake
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=torch.zeros((bs, 2), device=bev_queries.device).to(dtype),  # fake
            **kwargs
        )
        occ_pred = self.occ_proj(bev_embed)
        occ_pred = occ_pred.view(bs * seq_len, self.occ_xdim*self.occ_ydim, self.occ_zdim, self.occ_dims)
        occ_pred = occ_pred.permute(0, 2, 1, 3) # bs*seq_len, z, x*y, dim
        occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
        return occ_pred

    def upsample_occ(self, bev_for_occ, bs, seq_len):
        bev_for_occ = bev_for_occ.permute(1, 2, 0).contiguous().view(bs*seq_len, -1, self.bev_h, self.bev_w) # (bev_h*bev_w, bs*len_seq, c) -> (bs*len_seq, c, bev_h, bev_w)
        occ_pred = self.upsample_layer(bev_for_occ) # (bs*len_seq, c*occ_zdim, occ_xdim, occ_ydim)
        occ_pred = occ_pred.contiguous().view(bs*seq_len, self.occ_dims, self.occ_zdim, self.occ_xdim, self.occ_ydim) # (bs*len_seq, c, occ_zdim, occ_xdim, occ_ydim)
        occ_pred = occ_pred.permute(0, 2, 3, 4, 1) # (bs*len_seq, occ_zdim, bev_h, occ_ydim, c)
        occ_pred = occ_pred.contiguous().view(bs*seq_len, self.occ_zdim*self.occ_xdim*self.occ_ydim, self.occ_dims) # (bs*len_seq, occ_zdim*occ_xdim*occ_ydim, c)
        return occ_pred

    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        if self.transformer.decoder is not None:  # 3D detectio query
            object_query_embeds = self.query_embedding.weight.to(dtype)
        if not self.training:  # NOTE: Only difference to bevformer head
            object_query_embeds = object_query_embeds[:self.num_query // self.group_detr]
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        if isinstance(prev_bev, list) or isinstance(prev_bev, tuple):
            prev_bevs = [f.permute(1, 0, 2) for f in prev_bev]
            prev_bev = prev_bev[-1]
        elif torch.is_tensor(prev_bev) or prev_bev is None:
            prev_bevs = None
        else:
            raise NotImplementedError
        
        if only_bev:  # only use encoder to obtain BEV features
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        elif self.only_occ:
            bev_embed = self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
            # bev_embed: (bs, num_query, embed_dims)
            if prev_bevs is not None:
                bev_for_occ = torch.cat((*prev_bevs, bev_embed), dim=1)
                seq_len = len(prev_bevs) + 1
            else:
                bev_for_occ = bev_embed
                seq_len = 1
            
            if self.occ_head_type == 'mlp':
                if self.use_fine_occ:
                    occ_pred = self.occ_proj(bev_for_occ)
                    occ_pred = occ_pred.view(self.bev_h*self.bev_w, bs * seq_len, self.occ_zdim//2, self.occ_dims)
                    occ_pred = occ_pred.permute(1, 3, 2, 0)
                    occ_pred = occ_pred.view(bs * seq_len, self.occ_dims, self.occ_zdim//2, self.bev_h, self.bev_w)
                    occ_pred = self.up_sample(occ_pred)  # (bs*seq_len, C, occ_z, occ_y, occ_x)
                    occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
                    outputs_occupancy = self.occ_branches(occ_pred)
                    outputs_flow = None
                else:
                    occ_pred = self.occ_proj(bev_for_occ)
                    occ_pred = occ_pred.view(self.bev_h*self.bev_w, bs * seq_len, self.occ_zdim, self.occ_dims)
                    occ_pred = occ_pred.permute(1, 2, 0, 3) # bs*seq_len, z, x*y, dim
                    if self.with_occupancy_flow:
                        occ_pred = self.occupancy_aggregation(occ_pred.view(bs, seq_len, self.occ_zdim, self.occ_xdim, self.occ_ydim, self.occ_dims))

                    occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
                    outputs_occupancy = self.occ_branches(occ_pred)

                    if self.predict_flow:
                        outputs_flow = self.flow_branches(occ_pred)
                    else:
                        outputs_flow = None
                    
            elif self.occ_head_type == 'cnn':
                # bev_for_occ.shape: (bs, num_query, embed_dims)
                bev_for_occ = bev_for_occ.view(bs, 1, self.bev_h, self.bev_w, self.embed_dims)
                bev_for_occ = bev_for_occ.permute(0, 2, 3, 1, 4).flatten(3)  # (bs, bev_h, bev_w, -1)
                occ_pred = self.occ_proj(bev_for_occ)
                if self.use_fine_occ:
                    occ_pred = occ_pred.view(bs, self.bev_h, self.bev_w, self.occ_zdim//2, self.occ_dims)
                    occ_pred = occ_pred.permute(0, 4, 3, 1, 2)
                    occ_pred = self.up_sample(occ_pred)  # (bs, C, d, h, w)
                else:
                    occ_pred = occ_pred.view(bs, self.bev_h, self.bev_w, self.occ_zdim, self.occ_dims)
                    occ_pred = occ_pred.permute(0, 4, 3, 1, 2)  # (bs, occ_dims, z_dim, bev_h, bev_w)
                outputs_occupancy, outputs_flow = self.occ_seg_head(occ_pred)
                outputs_occupancy = outputs_occupancy.reshape(bs, self.occupancy_classes, -1)
                outputs_occupancy = outputs_occupancy.permute(0, 2, 1)
                if outputs_flow is not None:
                    outputs_flow = outputs_flow.reshape(bs, -1, 2)
            else:
                raise NotImplementedError

            # bev_embed = bev_embed.permute(1, 0, 2)  # (num_query, bs, embed_dims)
            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': None,
                'all_bbox_preds': None,
                'occupancy_preds': outputs_occupancy,
                'flow_preds': outputs_flow,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'enc_occupancy_preds': None
            }

            return outs

        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
                return_intermediate=True if self.occ_tsa else False
            )

        if self.occ_tsa:
            bev_embed, hs, init_reference, inter_references, feat_flatten, spatial_shapes, level_start_index = outputs
        else:
            bev_embed, hs, init_reference, inter_references = outputs

        # bev_embed: [bev_h*bev_w, bs, embed_dims]
        # hs:  (num_dec_layers, num_query, bs, embed_dims)
        # init_reference: (bs, num_query, 3)  in (0, 1)
        # inter_references:  (num_dec_layers, bs, num_query, 3)  in (0, 1)
        if prev_bevs is not None:
            bev_for_occ = torch.cat((*prev_bevs, bev_embed), dim=1)
            seq_len = len(prev_bevs) + 1
        else:
            bev_for_occ = bev_embed
            seq_len = 1
        if self.occ_tsa is None:
            occ_pred = self.upsample_occ(bev_for_occ, bs, seq_len) # bs*seq_len, z*x*y, num_classes
        else:
            occ_pred = self.upsample_tsa_occ(
                feat_flatten, 
                spatial_shapes, 
                level_start_index, 
                bev_for_occ, 
                bs, 
                seq_len,
                img_metas=img_metas)

        if self.with_occupancy_flow:
            occ_pred = self.occupancy_aggregation(occ_pred.view(bs, seq_len, self.occ_zdim, self.occ_xdim, self.occ_ydim, self.occ_dims))

        occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
        outputs_occupancy = self.occ_branches(occ_pred)

        if self.predict_flow:
            outputs_flow = self.flow_branches(occ_pred)
        else:
            outputs_flow = None

        # if self.with_color_render:
        #     outputs_color = self.color_branches(occ_pred)
        #     color_in_cams = self.voxel2image(outputs_color)
        #     occupancy_in_cams = self.voxel2image(outputs_occupancy)
        #     image_pred = self.render_image(color_in_cams, occupancy_in_cams)

        hs = hs.permute(0, 2, 1, 3)  # (num_dec_layers, bs, num_query, embed_dims)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'occupancy_preds': outputs_occupancy,
            'flow_preds': outputs_flow,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'enc_occupancy_preds': None
        }

        return outs