# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
from mmdet.models import HEADS
from third_party.bev_mmdet3d import BEVFormerOccupancyHeadApollo
from mmdet.models.utils.transformer import inverse_sigmoid
from ..utils import LINEAR_LAYERS


class ReLUAddZeros(nn.Module):
    def __init__(self):
        super(ReLUAddZeros, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + torch.zeros_like(x))


@HEADS.register_module()
class BEVFormerOccupancyHeadApolloTRT(BEVFormerOccupancyHeadApollo):
    """Head of BEVFormerOccupancyHead.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 linear_cfg=None,
                 **kwargs):
        if linear_cfg is None:
            linear_cfg = dict(type="Linear")
        self.linear = LINEAR_LAYERS.get(linear_cfg["type"])
        super(BEVFormerOccupancyHeadApolloTRT, self).__init__(
            *args, **kwargs)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        if self.transformer.decoder is not None:
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(self.linear(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(self.embed_dims))
                cls_branch.append(nn.ReLU(inplace=True))
            cls_branch.append(self.linear(self.embed_dims, self.cls_out_channels))
            fc_cls = nn.Sequential(*cls_branch)

            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(self.linear(self.embed_dims, self.embed_dims))
                reg_branch.append(
                    nn.ReLU(inplace=True) if self.linear == nn.Linear else ReLUAddZeros()
            )
            reg_branch.append(self.linear(self.embed_dims, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)

            def _get_clones(module, N):
                return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

            # last reg_branch is used to generate proposal from
            # encode feature map when as_two_stage is True.
            num_pred = (self.transformer.decoder.num_layers + 1) if \
                self.as_two_stage else self.transformer.decoder.num_layers

            if self.with_box_refine:
                self.cls_branches = _get_clones(fc_cls, num_pred)
                self.reg_branches = _get_clones(reg_branch, num_pred)
            else:
                self.cls_branches = nn.ModuleList(
                    [fc_cls for _ in range(num_pred)])
                self.reg_branches = nn.ModuleList(
                    [reg_branch for _ in range(num_pred)])

            if not self.as_two_stage:
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims)
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
        else:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        if self.use_fine_occ:
            self.occ_proj = self.linear(self.embed_dims, self.occ_dims * self.occ_zdim//2)
        else:
            self.occ_proj = self.linear(self.embed_dims, self.occ_dims * self.occ_zdim)

        # occupancy branch
        occ_branch = []
        for _ in range(self.num_occ_fcs):
            occ_branch.append(self.linear(self.occ_dims, self.occ_dims))
            occ_branch.append(nn.LayerNorm(self.occ_dims))
            occ_branch.append(nn.ReLU(inplace=True))
        occ_branch.append(self.linear(self.occ_dims, self.occupancy_classes))
        self.occ_branches = nn.Sequential(*occ_branch)
        
        # upsampling branch
        if self.use_fine_occ:
            self.up_sample = nn.Upsample(size=(self.occ_zdim, self.occ_ydim, self.occ_xdim), mode='trilinear', align_corners=True)

        # flow branch: considering the foreground objects
        # predict flow for each class separately  TODO
        if self.loss_flow is not None:
            flow_branch = []
            for _ in range(self.num_occ_fcs):
                flow_branch.append(self.linear(self.occ_dims, self.occ_dims))
                flow_branch.append(nn.LayerNorm(self.occ_dims))
                flow_branch.append(nn.ReLU(inplace=True))
            flow_branch.append(self.linear(self.occ_dims, self.flow_gt_dimension))
            self.flow_branches = nn.Sequential(*flow_branch)

        if self.with_occupancy_flow:
            self.forward_flow = nn.Linear(self.occ_dims, 3)
            self.backward_flow = nn.Linear(self.occ_dims, 3)

            flow_fc = []
            for _ in range(self.num_occ_fcs):
                flow_fc.append(self.linear(self.occ_dims, self.occ_dims))
                flow_fc.append(nn.LayerNorm(self.occ_dims))
                flow_fc.append(nn.ReLU(inplace=True))
            self.flow_fc = nn.Sequential(*flow_fc)

        if self.with_color_render:
            color_branch = []
            for _ in range(self.num_occ_fcs):
                color_branch.append(self.linear(self.occ_dims, self.occ_dims))
                color_branch.append(nn.LayerNorm(self.occ_dims))
                color_branch.append(nn.ReLU(inplace=True))
            color_branch.append(self.linear(self.occ_dims, 3))
            self.color_branches = nn.Sequential(*color_branch)

    def upsample_tsa_occ_trt(
            self,
            feat_flatten,
            spatial_shapes,
            level_start_index,
            lidar2img,
            image_shape,
            bev_for_occ,
            bs,
            seq_len):
        bev_for_occ = bev_for_occ.permute(1, 2, 0).contiguous().view(bs*seq_len, -1, self.bev_h, self.bev_w)
        upsampled_bev_embed = self.upsample_layer(bev_for_occ)
        bev_queries = upsampled_bev_embed.flatten(2).permute(2, 0, 1)
        dtype = feat_flatten[0].dtype
        occ_bev_mask = torch.zeros((bs, self.occ_xdim, self.occ_ydim),
                                    device=bev_queries.device).to(dtype)
        query_pos = self.positional_encoding_occ(occ_bev_mask).to(dtype)
        query_pos = query_pos.flatten(2).permute(0, 2, 1)
        bev_embed = self.occ_tsa.forward_trt(
            bev_queries,
            feat_flatten,
            feat_flatten,
            query_pos,
            lidar2img=lidar2img,
            bev_h=self.occ_xdim,
            bev_w=self.occ_ydim,
            bev_pos=torch.zeros_like(bev_queries).permute(1, 0, 2),  # fake
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=torch.zeros_like(bev_queries),  # fake
            shift=torch.zeros((bs, 2), device=bev_queries.device).to(dtype),  # fake
            image_shape=image_shape,
            use_prev_bev=torch.tensor([0.0], device=bev_queries.device).to(dtype),
        )
        occ_pred = self.occ_proj(bev_embed)
        occ_pred = occ_pred.view(bs * seq_len, self.occ_xdim*self.occ_ydim, self.occ_zdim, self.occ_dims)
        occ_pred = occ_pred.permute(0, 2, 1, 3)
        occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
        return occ_pred

    def forward_trt(
            self, mlvl_feats, prev_bev, can_bus, lidar2img, image_shape, use_prev_bev
        ):

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        if self.transformer.decoder is not None:
            object_query_embeds = self.query_embedding.weight.to(dtype)
        if not self.training:
            object_query_embeds = object_query_embeds[:self.num_query // self.group_detr]
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if self.transformer.decoder is None:
            bev_embed = self.transformer.get_bev_features_trt(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                can_bus,
                lidar2img,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                image_shape=image_shape,
                use_prev_bev=use_prev_bev,
            )
        else:
            outputs = self.transformer.forward_trt(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches
                if self.with_box_refine
                else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                can_bus=can_bus,
                lidar2img=lidar2img,
                prev_bev=prev_bev,
                image_shape=image_shape,
                use_prev_bev=use_prev_bev,
                return_intermediate=True if self.occ_tsa else False
            )

            if self.occ_tsa:
                bev_embed, hs, init_reference, inter_references, feat_flatten, spatial_shapes, level_start_index = outputs
            else:
                bev_embed, hs, init_reference, inter_references = outputs

            hs = hs.permute(0, 2, 1, 3)
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
                tmp[..., 0:1] = (
                    tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                )
                tmp[..., 1:2] = (
                    tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
                )
                tmp[..., 4:5] = (
                    tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
                )

                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)

        bev_for_occ = bev_embed
        seq_len = 1
        if self.occ_tsa is None:
            occ_pred = self.upsample_occ(bev_for_occ, bs, seq_len)
        else:
            occ_pred = self.upsample_tsa_occ_trt(
                feat_flatten,
                spatial_shapes,
                level_start_index,
                lidar2img,
                image_shape,
                bev_for_occ,
                bs,
                seq_len)

        if self.with_occupancy_flow:
            occ_pred = self.occupancy_aggregation(occ_pred.view(bs, seq_len, self.occ_zdim, self.occ_xdim, self.occ_ydim, self.occ_dims))

        occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
        outputs_occupancy = self.occ_branches(occ_pred)

        # bev_embed = bev_embed.permute(1, 0, 2)  # (num_query, bs, embed_dims)
        if self.transformer.decoder is not None:
            return bev_embed, outputs_classes, outputs_coords, outputs_occupancy
        else:
            return bev_embed, outputs_occupancy

@HEADS.register_module()
class BEVFormerOccupancyHeadApolloTRTP(BEVFormerOccupancyHeadApolloTRT):
    def __init__(self, *args, **kwargs):
        super(BEVFormerOccupancyHeadApolloTRTP, self).__init__(*args, **kwargs)

    def forward_trt(
            self, mlvl_feats, prev_bev, can_bus, lidar2img, image_shape, use_prev_bev
        ):

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        if self.transformer.decoder is not None:
            object_query_embeds = self.query_embedding.weight.to(dtype)
        if not self.training:
            object_query_embeds = object_query_embeds[:self.num_query // self.group_detr]
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if self.transformer.decoder is None:
            bev_embed = self.transformer.get_bev_features_trt(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                can_bus,
                lidar2img,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                image_shape=image_shape,
                use_prev_bev=use_prev_bev,
            )
        else:
            outputs = self.transformer.forward_trt(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches
                if self.with_box_refine
                else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                can_bus=can_bus,
                lidar2img=lidar2img,
                prev_bev=prev_bev,
                image_shape=image_shape,
                use_prev_bev=use_prev_bev,
                return_intermediate=True if self.occ_tsa else False
            )

            if self.occ_tsa:
                bev_embed, hs, init_reference, inter_references, feat_flatten, spatial_shapes, level_start_index = outputs
            else:
                # bev_embed, hs, init_reference, inter_references = outputs
                bev_embed, hs, init_reference, inter_references, bev_query_input, img_feats_input, bev_pos_input, hybird_ref_2d, ref_3d, reference_points_cam, bev_mask_encoder, bev_embed_one, bev_embed_two = outputs

            init_reference = init_reference.view(1, self.num_query // self.group_detr, 3)
            inter_references = inter_references.view(-1, 1, self.num_query // self.group_detr, 3)
            hs = hs.view(-1, 1, self.num_query // self.group_detr, self.embed_dims)
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
                tmp[..., 0:1] = (
                    tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                )
                tmp[..., 1:2] = (
                    tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
                )
                tmp[..., 4:5] = (
                    tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
                )

                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)

        bev_for_occ = bev_embed
        seq_len = 1
        if self.occ_tsa is None:
            occ_pred = self.upsample_occ(bev_for_occ, bs, seq_len)
        else:
            occ_pred = self.upsample_tsa_occ_trt(
                feat_flatten,
                spatial_shapes,
                level_start_index,
                lidar2img,
                image_shape,
                bev_for_occ,
                bs,
                seq_len)

        if self.with_occupancy_flow:
            occ_pred = self.occupancy_aggregation(occ_pred.view(bs, seq_len, self.occ_zdim, self.occ_xdim, self.occ_ydim, self.occ_dims))

        occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
        outputs_occupancy = self.occ_branches(occ_pred)

        # bev_embed = bev_embed.permute(1, 0, 2)  # (num_query, bs, embed_dims)
        if self.transformer.decoder is not None:
            # return bev_embed, outputs_classes, outputs_coords, outputs_occupancy
            return bev_embed, outputs_classes, outputs_coords, outputs_occupancy, bev_query_input, img_feats_input, bev_pos_input, hybird_ref_2d, ref_3d, reference_points_cam, bev_mask_encoder, bev_embed_one, bev_embed_two
        else:
            return bev_embed, outputs_occupancy