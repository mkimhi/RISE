import numpy as np
import torch


class visual_aid():
    def __init__(self):
        super().__init__()
        self.PSEUDO_OBJ_ACC = []
        self.PSEUDO_MASK_ACC = []
        self.NUM_SAVED_IMGS=0
        
    def update_obj_mask_acc(self,det_targets,pseudo_det_targets):
        mask_acc = calc_mask_acc(det_targets,pseudo_det_targets)
        num_obj_acc = len(pseudo_det_targets[0]['labels'])/len(det_targets[0]['labels'])
        self.PSEUDO_OBJ_ACC.append(num_obj_acc)
        self.PSEUDO_MASK_ACC.append(mask_acc)
        
    def save_obj_mask_acc(self):    
        np.save(f'obj_acc.npy', np.array(self.PSEUDO_OBJ_ACC))
        np.save(f'mask_acc.npy', np.array(self.PSEUDO_MASK_ACC))

    def save_img_gt_labels(self,images,gt_instances,pseudo_instances):
        self.NUM_SAVED_IMGS += 1
        torch.save(images.tensor[0].cpu(),f'imgs/image_{self.NUM_SAVED_IMGS}_0')
        torch.save(gt_instances[0].get('gt_masks').tensor.cpu(),f'imgs/gt_{self.NUM_SAVED_IMGS}_0')
        torch.save(pseudo_instances[0].get('gt_masks').tensor.cpu(),f'imgs/pred_{self.NUM_SAVED_IMGS}_0')
        if len(images.tensor.shape) > 1:
            torch.save(images.tensor[1].cpu(),f'imgs/image_{self.NUM_SAVED_IMGS}_1')
            torch.save(gt_instances[1].get('gt_masks').tensor.cpu(),f'imgs/gt_{self.NUM_SAVED_IMGS}_1')
            torch.save(pseudo_instances[1].get('gt_masks').tensor.cpu(),f'imgs/pred_{self.NUM_SAVED_IMGS}_1')
    def save_precentile(self,max_score,th_decay_ratio):
        timestamp = int(th_decay_ratio*10000)
        nintyfive = np.percentile(max_score.detach().cpu().numpy().flatten(), 95)
        nintynine = np.percentile(max_score.detach().cpu().numpy().flatten(), 99)
        nintyninefive = np.percentile(max_score.detach().cpu().numpy().flatten(), 99.5)
        torch.save(torch.tensor(nintyfive),f'box_95th_precentile_{timestamp}.pt')
        torch.save(torch.tensor(nintynine),f'box_99th_precentile_{timestamp}.pt')
        torch.save(torch.tensor(nintyninefive),f'box_995th_precentile_{timestamp}.pt')


def calc_mask_acc(target,pseudo_target):
    if target[0]['masks'].shape[0] == pseudo_target[0]['masks'].shape[0]:
        total_shape = target[0]['masks'].shape[0] * target[0]['masks'].shape[1] * target[0]['masks'].shape[2]
        mask_acc = (target[0]['masks'] == pseudo_target[0]['masks']).sum() / total_shape
    else:
        n_obj = min(target[0]['masks'].shape[0], pseudo_target[0]['masks'].shape[0])
        total_shape = n_obj * target[0]['masks'].shape[1] * target[0]['masks'].shape[2]
        mask_acc = (target[0]['masks'][:n_obj] == pseudo_target[0]['masks'][:n_obj]).sum() / total_shape
    return float(mask_acc)
