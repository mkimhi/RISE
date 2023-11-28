
#general
import math
import numpy as np
import copy
from typing import Dict, List, Union
from PIL import Image
import random

#torch and torchvision
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as TVT
import torchvision.transforms.functional as TF
import torchvision.ops as ops

#internal and detectron:
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.data import transforms as T
from fvcore.nn import giou_loss, smooth_l1_loss
from .models.backbone import Joiner
from .models.deformable_detr import DeformableDETR, SetCriterion
from .models.matcher import HungarianMatcher
from .models.position_encoding import PositionEmbeddingSine
from .models.deformable_transformer import DeformableTransformer
from .models.segmentation_condInst import CondInst_segm, segmentation_postprocess
from .models.tracker import Tracker
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import NestedTensor
from .util.cut_paste import CopyPasteBank,image_obj,tv_save_image,save_imgs
from .data.augmentation import create_strong_images
__all__ = ["RISE"]

class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = [backbone_shape[f].channels for f in backbone_shape.keys()]
       
    def forward(self, tensor_list):
        xs = self.backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks



@META_ARCH_REGISTRY.register()
class RISE(nn.Module):
    """
    Implement RISE
    """

    def __init__(self, cfg):
        super().__init__()
        self.NUM=0
        augs = List[Union[T.Augmentation, T.Transform]]
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.clip_stride = cfg.MODEL.RISE.CLIP_STRIDE

        ### inference setting
        self.merge_on_cpu = cfg.MODEL.RISE.MERGE_ON_CPU
        self.is_multi_cls = cfg.MODEL.RISE.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.RISE.APPLY_CLS_THRES
        self.temporal_score_type = cfg.MODEL.RISE.TEMPORAL_SCORE_TYPE
        self.inference_select_thres = cfg.MODEL.RISE.INFERENCE_SELECT_THRES
        self.inference_fw = cfg.MODEL.RISE.INFERENCE_FW
        self.inference_tw = cfg.MODEL.RISE.INFERENCE_TW
        self.memory_len = cfg.MODEL.RISE.MEMORY_LEN
        self.nms_pre = cfg.MODEL.RISE.NMS_PRE
        self.add_new_score = cfg.MODEL.RISE.ADD_NEW_SCORE 
        self.batch_infer_len = cfg.MODEL.RISE.BATCH_INFER_LEN


        self.is_coco = cfg.DATASETS.TEST[0].startswith("coco")
        self.num_classes = cfg.MODEL.RISE.NUM_CLASSES
        self.mask_stride = cfg.MODEL.RISE.MASK_STRIDE
        self.match_stride = cfg.MODEL.RISE.MATCH_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON

        self.coco_pretrain = cfg.INPUT.COCO_PRETRAIN
        hidden_dim = cfg.MODEL.RISE.HIDDEN_DIM
        num_queries = cfg.MODEL.RISE.NUM_OBJECT_QUERIES

        ##semi params
        self.semi = cfg.SEMI  #enables the use of unlabled data
        self.cls_th= cfg.CLF_TH
        self.mask_th = cfg.MASK_TH
        self.dpa = cfg.DPA
        self.vis = cfg.VIS_OT#if to visualize optimal transport

        # Loss parameters:
        mask_weight = cfg.MODEL.RISE.MASK_WEIGHT
        dice_weight = cfg.MODEL.RISE.DICE_WEIGHT
        giou_weight = cfg.MODEL.RISE.GIOU_WEIGHT
        l1_weight = cfg.MODEL.RISE.L1_WEIGHT
        class_weight = cfg.MODEL.RISE.CLASS_WEIGHT
        reid_weight = cfg.MODEL.RISE.REID_WEIGHT
        deep_supervision = cfg.MODEL.RISE.DEEP_SUPERVISION

        focal_alpha = cfg.MODEL.RISE.FOCAL_ALPHA

        set_cost_class = cfg.MODEL.RISE.SET_COST_CLASS
        set_cost_bbox = cfg.MODEL.RISE.SET_COST_BOX
        set_cost_giou = cfg.MODEL.RISE.SET_COST_GIOU


        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels[1:]  # only take [c3 c4 c5] from resnet and gengrate c6 later
        backbone.strides = d2_backbone.feature_strides[1:]

        
        transformer = DeformableTransformer(
        d_model= hidden_dim,
        nhead=cfg.MODEL.RISE.NHEADS,
        num_encoder_layers=cfg.MODEL.RISE.ENC_LAYERS,
        num_decoder_layers=cfg.MODEL.RISE.DEC_LAYERS,
        dim_feedforward=cfg.MODEL.RISE.DIM_FEEDFORWARD,
        dropout=cfg.MODEL.RISE.DROPOUT,
        activation="relu",
        return_intermediate_dec=True,
        num_frames=self.num_frames,
        num_feature_levels=cfg.MODEL.RISE.NUM_FEATURE_LEVELS,
        dec_n_points=cfg.MODEL.RISE.DEC_N_POINTS,
        enc_n_points=cfg.MODEL.RISE.ENC_N_POINTS,)
        
        
        model = DeformableDETR(
        backbone,
        transformer,
        num_classes=self.num_classes,
        num_frames=self.num_frames,
        num_queries=num_queries,
        num_feature_levels=cfg.MODEL.RISE.NUM_FEATURE_LEVELS,
        aux_loss=deep_supervision,
        with_box_refine=True )

        self.detr = CondInst_segm(model, freeze_detr=False, rel_coord=True )
        
        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(multi_frame=True,
                            cost_class=set_cost_class,
                            cost_bbox=set_cost_bbox,
                            cost_giou=set_cost_giou)

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou":giou_weight}
        weight_dict["loss_reid"] = reid_weight
        weight_dict["loss_reid_aux"] = reid_weight*1.5
        weight_dict["loss_mask"] = mask_weight
        weight_dict["loss_dice"] = dice_weight

        if deep_supervision:
            aux_weight_dict = {}
            for i in range(cfg.MODEL.RISE.DEC_LAYERS - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
  
        losses = ['labels', 'boxes', 'masks','reid']
        

        self.criterion = SetCriterion(self.num_classes, matcher, weight_dict, losses, 
                             mask_out_stride=self.mask_stride,
                             focal_alpha=focal_alpha,
                             num_frames = self.num_frames)
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.merge_device = "cpu" if self.merge_on_cpu else self.device
        self.cp_bank = CopyPasteBank()
        self.rotation = TVT.RandomRotation(4)
        self.prespective = TVT.RandomPerspective(distortion_scale=0.3)
    def forward(self, batched_inputs, labeled=True,th_decay_ratio=1):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
            labeled: if False, each item of batched_inputs has no 'instances' 
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:

            if labeled:
                images = self.preprocess_image(batched_inputs)
                gt_instances = []
                for i,video in enumerate(batched_inputs):
                    gt_instances.append(video["instances"][0].to(self.device))
                    image = images[i]
                    gt_ins = gt_instances[i]
                    gt_instances[i],n_objects = self.cp_bank.paste(image, gt_ins)
                    gt_instances.append(gt_instances[-1])
                    
                    #reference frame only
                    image = images[i+1]
                    gt_ins = gt_instances[i+1]
                    image = self.rotation(image)#small image pertubation
                    gt_ins.gt_masks.tensor = self.rotation(gt_ins.gt_masks.tensor)                    
                    
                    gt_instances[i+1], _count = self.cp_bank.paste(image, gt_ins, n_objects=1)

                self.cp_bank.append(images,gt_instances)
                det_targets,ref_targets = self.prepare_targets(gt_instances) #return instances target ids
                
                output, loss_dict = self.detr(images, det_targets,ref_targets, self.criterion, train=True)
                
            else: #create pseudo labels
                ## prepare weak images for pseudo-labels
                images_w= self.preprocess_image(batched_inputs)#.to(self.device)
                with torch.no_grad():
                    self.detr.eval()
                    u_preds_w = self.detr.inference_forward(images_w)

                #validation
                gt_instances = []
                for video in batched_inputs:
                    for frame in video["instances"]:
                        gt_instances.append(frame.to(self.device))
                    if len(self.cp_bank) > 2:
                        image = images_w[len(gt_instances)-1]
                        gt_instance = gt_instances[-1]
                        gt_instances[-1],_ = self.cp_bank.paste(image, gt_instance, n_objects=1)
                det_targets,ref_targets = self.prepare_targets(gt_instances)

                pseudo_instances =  self.generate_pseudo_instances(u_preds_w,batched_inputs,images_w,th_decay_ratio,gt_instances = gt_instances)
                pseudo_det_targets,pseudo_ref_targets = self.prepare_targets(pseudo_instances) #return instances target ids
                if pseudo_det_targets[0]['valid'].sum() < 2 or pseudo_ref_targets[0]['valid'].sum() < 2:
                    pseudo_det_targets,pseudo_ref_targets = det_targets,ref_targets

                images_s= create_strong_images(images_w)
                self.detr.train()
                output, loss_dict = self.detr(images_s, pseudo_det_targets,pseudo_ref_targets, self.criterion, train=True)
                #No supervision signal
                if pseudo_det_targets[0]['valid'].sum()==0 or pseudo_ref_targets[0]['valid'].sum()==0:
                    pass
            
            weight_dict  = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k] * 0.1
                # if 'reid' in k:
                #     loss_dict[k] *=.05
                # if 'SEMI' in k and 'dice' in k:
                #     loss_dict[k]*=.05
            return loss_dict
        else:
            raise NotImplementedError

    def generate_pseudo_instances(self,u_preds_w,batched_inputs,images_w,th_decay_ratio,gt_instances,M2B=True,MLM=True):
        psuedo_instances = self.create_dummy_instances(batched_inputs)
        for i in range(len(u_preds_w['pred_logits'])): #iterate over images
            logits = u_preds_w['pred_logits'][i]
            output_boxes = u_preds_w['pred_boxes'][i]
            image_sizes = images_w.image_sizes[i]
            scores = logits.sigmoid().cpu().detach()  
            max_score, _ = torch.max(logits.sigmoid(),1)
            thresh = self.cls_th
            if self.dpa: #DPA decay
                th_decay_ratio = min(th_decay_ratio,0.995)
                percent = self.cls_th * 100 * ((1 - th_decay_ratio)**.1)
                thresh = np.percentile(max_score.detach().cpu().numpy().flatten(), percent)
                thresh = max(thresh,0.85) #bounded
            
            indices = torch.nonzero(max_score > thresh, as_tuple=False).squeeze(1)
            if not len(indices):
                return gt_instances

            if not MLM:
                nms_scores,idxs = torch.max(logits.sigmoid()[indices],1)
                boxes_before_nms = box_cxcywh_to_xyxy(output_boxes[indices])
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.9)#.tolist()
                indices = indices[keep_indices]
            
            box_score,_ = torch.max(logits.sigmoid()[indices],1)
            
            scale_mask = torch.tensor([image_sizes[0],image_sizes[1],image_sizes[0],image_sizes[1]])
            pred_boxes = box_cxcywh_to_xyxy(output_boxes[indices])*scale_mask.clone().detach().to(self.device).type(torch.int).type(torch.float)

            
            #save for tracker
            det_bboxes = torch.cat([output_boxes[indices],box_score.unsqueeze(1)],dim=1) #objects x 5 (box,score) 
            det_labels = torch.argmax(logits.sigmoid()[indices].unsqueeze(0),dim=2) #objects x 1 
            if len(det_labels.shape) == 2:
                det_labels = det_labels.squeeze(1)

            output_mask = u_preds_w['pred_masks'][i]
            output_mask=output_mask.to(self.device)
            mask_i = output_mask[indices].squeeze(1) #objects x x h x w 

            output_h, output_w = mask_i.shape[-2:] 
            pseudo_mask_i =F.interpolate(mask_i[:,None,:,:],  size=(output_h*4, output_w*4) ,mode="bilinear", align_corners=False).sigmoid()
            pseudo_mask_i = pseudo_mask_i[:,:,:image_sizes[0],:image_sizes[1]] #crop the padding area
            pseudo_mask_i = (F.interpolate(pseudo_mask_i, size=(image_sizes[0], image_sizes[1]), mode='nearest')>self.mask_th)[:,0]           
    
            if M2B:
                for j,obj_mask in enumerate(pseudo_mask_i):
                    obj_cords = torch.nonzero(obj_mask) #check for more then one obj
                    x1,y1 = obj_cords.min(0)[0]
                    x2,y2 = obj_cords.max(0)[0]
                    pred_boxes[j] = torch.tensor((x1,y1,x2,y2))

            #assign pseudo instance
            psuedo_instances[i]._image_size = image_sizes
            psuedo_instances[i]._fields['gt_boxes'].tensor = pred_boxes
            psuedo_instances[i]._fields['gt_classes'] = det_labels #obj
            psuedo_instances[i]._fields['gt_masks'].tensor = pseudo_mask_i #objxhxw
            psuedo_instances[i]._fields['gt_masks'].image_size = image_sizes #hxw
            psuedo_instances[i]._fields['gt_ids'] = psuedo_instances[i]._fields['gt_ids'][:len(det_labels)]

        return psuedo_instances

    def create_dummy_instances(self,batched_inputs,to_paste=False):
        psuedo_instances = []
        for video in batched_inputs: #video['image] list of two
            for frame in video["instances"]:
                psuedo_instances.append(frame.to(self.device))
            if to_paste and len(self.cp_bank) > 2:
                images_w= self.preprocess_image(batched_inputs)
                image = images_w[len(psuedo_instances)-1]
                gt_instance = psuedo_instances[-1]
                psuedo_instances[-1],n_objects = self.cp_bank.paste(image, gt_instance, n_objects=1)
        return psuedo_instances

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            if len(gt_classes.shape)==2:
                gt_classes = gt_classes.squeeze(0)
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            gt_masks = targets_per_image.gt_masks.tensor
            inst_ids = targets_per_image.gt_ids
            valid_id = inst_ids!=-1  # if a object is disappeared，its gt_ids is -1 
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, 'inst_id':inst_ids, 'valid':valid_id})
        bz = len(new_targets)//2
        key_ids = list(range(0,bz*2-1,2))
        ref_ids = list(range(1,bz*2,2))
        det_targets = [new_targets[_i] for _i in key_ids]
        ref_targets = [new_targets[_i] for _i in ref_ids]
        
        for i in range(bz):  # fliter empety object in key frame
            det_target = det_targets[i] #shape: [num_obj, 4]
            ref_target = ref_targets[i] #shape: [num_obj, 4]
            if False in det_target['valid']:
                valid_i = det_target['valid'].clone()
                for k,v in det_target.items():
                    if len(v)!=len(valid_i):
                        if len(valid_i)>len(v):
                            valid_i = valid_i[:len(v)]
                        else:    
                            residual_obj = torch.ones(len(v)-len(valid_i)).to(v)
                            valid_i = torch.cat((valid_i,residual_obj))
                    det_target[k] = v[valid_i]
                for k,v in ref_target.items():
                    if len(v)!=len(valid_i):
                        if len(valid_i)>len(v):
                            valid_i = valid_i[:len(v)]
                        else:
                            residual_obj = torch.ones(len(v)-len(valid_i)).to(v)
                            valid_i = torch.cat((valid_i,residual_obj))
                    ref_target[k] = v[valid_i]

 
        return det_targets,ref_targets

    def inference(self, outputs, tracker, ori_size, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # results = []
        video_dict = {}
        vido_logits = outputs['pred_logits']
        video_output_masks = outputs['pred_masks']
        output_h, output_w = video_output_masks.shape[-2:]
        video_output_boxes = outputs['pred_boxes']
        video_output_embeds = outputs['pred_inst_embed']
        vid_len = len(vido_logits)
        #self.inference_select_thres=0
        for i_frame, (logits, output_mask, output_boxes, output_embed) in enumerate(zip(
            vido_logits, video_output_masks, video_output_boxes, video_output_embeds
         )):
            scores = logits.sigmoid().cpu().detach()  #[300,42]
            max_score, _ = torch.max(logits.sigmoid(),1)
            indices = torch.nonzero(max_score>self.inference_select_thres, as_tuple=False).squeeze(1)
            if len(indices) == 0:
                topkv, indices_top1 = torch.topk(scores.max(1)[0],k=1)
                indices_top1 = indices_top1[torch.argmax(topkv)]
                indices = [indices_top1.tolist()]
            else:
                nms_scores,idxs = torch.max(logits.sigmoid()[indices],1)
                boxes_before_nms = box_cxcywh_to_xyxy(output_boxes[indices])
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.9)#.tolist()
                indices = indices[keep_indices]
            box_score = torch.max(logits.sigmoid()[indices],1)[0]
            det_bboxes = torch.cat([output_boxes[indices],box_score.unsqueeze(1)],dim=1)
            det_labels = torch.argmax(logits.sigmoid()[indices],dim=1)
            track_feats = output_embed[indices]
            output_mask=output_mask.to(self.device)
            det_masks = output_mask[indices]

            bboxes, labels, ids, indices = tracker.match(
            bboxes=det_bboxes,
            labels=det_labels,
            masks = det_masks,
            track_feats=track_feats,
            frame_id=i_frame,
            indices = indices)
            indices = torch.tensor(indices)[ids>-1].tolist()
            ids = ids[ids > -1]
            ids = ids.tolist()
            for query_i, id in zip(indices,ids):
                if id in video_dict.keys():
                    video_dict[id]['masks'].append(output_mask[query_i])
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1
                else:
                    video_dict[id] = {
                        'masks':[None for fi in range(i_frame)], 
                        'boxes':[None for fi in range(i_frame)], 
                        'scores':[None for fi in range(i_frame)], 
                        'valid':0}
                    video_dict[id]['masks'].append(output_mask[query_i])
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1

            for k,v in video_dict.items():
                if len(v['masks'])<i_frame+1: #padding None for unmatched ID
                    v['masks'].append(None)
                    v['scores'].append(None)
                    v['boxes'].append(None)
            check_len = [len(v['masks']) for k,v in video_dict.items()]
            # print('check_len',check_len)

            #  filtering sequences that are too short in video_dict (noise)，the rule is: if the first two frames are None and valid is less than 3
            if i_frame>8:
                del_list = []
                for k,v in video_dict.items():
                    if v['masks'][-1] is None and  v['masks'][-2] is None and v['valid']<3:
                        del_list.append(k)   
                for del_k in del_list:
                    video_dict.pop(del_k)                      

        del outputs
        logits_list = []
        masks_list = []

        for inst_id,m in  enumerate(video_dict.keys()):
            score_list_ori = video_dict[m]['scores']
            scores_temporal = []
            for k in score_list_ori:
                if k is not None:
                    scores_temporal.append(k)
            logits_i = torch.stack(scores_temporal)
            if self.temporal_score_type == 'mean':
                logits_i = logits_i.mean(0)
            elif self.temporal_score_type == 'max':
                logits_i = logits_i.max(0)[0]
            else:
                print('non valid temporal_score_type')
                import sys;sys.exit(0)
            logits_list.append(logits_i)
            
            # category_id = np.argmax(logits_i.mean(0))
            masks_list_i = []
            for n in range(vid_len):
                mask_i = video_dict[m]['masks'][n] #shape: [num_obj, 4]
                if mask_i is None:    
                    zero_mask = None # padding None instead of zero mask to save memory
                    masks_list_i.append(zero_mask)
                else:
                    mask_size = (output_h*4, output_w*4)
                    pred_mask_i =F.interpolate(mask_i[:,None,:,:],size=mask_size ,mode="bilinear", align_corners=False).sigmoid()
                    pred_mask_i = pred_mask_i[:,:,:image_sizes[0],:image_sizes[1]] #crop the padding area
                    pred_mask_i = (F.interpolate(pred_mask_i, size=(ori_size[0], ori_size[1]), mode='nearest')>0.5)[0,0].cpu() # resize to ori video size
                    masks_list_i.append(pred_mask_i)
            masks_list.append(masks_list_i)
        if len(logits_list)>0:
            pred_cls = torch.stack(logits_list)
        else:
            pred_cls = []

        if len(pred_cls) > 0:
            if self.is_multi_cls:
                is_above_thres = torch.where(pred_cls > self.apply_cls_thres)
                scores = pred_cls[is_above_thres]
                labels = is_above_thres[1]
                out_masks = [masks_list[valid_id] for valid_id in is_above_thres[0]]
            else:
                scores, labels = pred_cls.max(-1)
                out_masks = masks_list
            out_scores = scores.tolist()
            out_labels = labels.tolist()
        else:
            out_scores = []
            out_labels = []
            out_masks = []
        video_output = {
            "image_size": ori_size,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output


    def preprocess_image(self, batched_inputs,labeled=True):
        """
        Normalize, pad and batch the input images.
        """
        if labeled:
            images = []
            for video in batched_inputs:
                for frame in video["image"]:
                    images.append(self.normalizer(frame.to(self.device)))
            images = ImageList.from_tensors(images)
            return images
        else:
            images,images_s = [],[]
            for video in batched_inputs:
                print(video[0][0]['image'][0].shape)
                for frame in video[0][0]['image']:
                    images.append(self.normalizer(frame.to(self.device)))
                for frame in video[1][0]['image']:
                    images_s.append(self.normalizer(frame.to(self.device)))
            images = ImageList.from_tensors(images)
            images_s = ImageList.from_tensors(images_s)
            return images,images_s



    def coco_inference(self, box_cls, box_pred, mask_pred, image_sizes):
      
        assert len(box_cls) == len(image_sizes)
        results = []

        for i, (logits_per_image, box_pred_per_image, image_size) in enumerate(zip(
            box_cls, box_pred, image_sizes
        )):



            prob = logits_per_image.sigmoid()
            nms_scores,idxs = torch.max(prob,1)
            boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
            keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.7)  
            prob = prob[keep_indices]
            box_pred_per_image = box_pred_per_image[keep_indices]
            mask_pred_i = mask_pred[i][keep_indices]

            topk_values, topk_indexes = torch.topk(prob.view(-1), 100, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
            # topk_boxes = topk_indexes // logits_per_image.shape[1]
            labels = topk_indexes % logits_per_image.shape[1]
            scores_per_image = scores
            labels_per_image = labels

            box_pred_per_image = box_pred_per_image[topk_boxes]
            mask_pred_i = mask_pred_i[topk_boxes]

            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                N, C, H, W = mask_pred_i.shape
                mask = F.interpolate(mask_pred_i, size=(H*4, W*4), mode='bilinear', align_corners=False)
                mask = mask.sigmoid() > 0.5
                mask = mask[:,:,:image_size[0],:image_size[1]]
                result.pred_masks = mask

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_coco_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


