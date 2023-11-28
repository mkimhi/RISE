from torchvision.utils import save_image
from collections import deque
from random import sample, randrange
import numpy as np
import torch


class image_obj():
    def __init__(self,obj_masked,mask,obj_class,obj_id,obj_shape):
        super().__init__()
        self.obj_masked = obj_masked #3 x w x h
        self.mask = mask #w x h
        self.obj_class = obj_class #int
        self.obj_id = obj_id #int
        self.obj_shape = obj_shape #tuple (w,h)
    def get_shape(self):
        return self.obj_shape #tuple (w,h)
    def get_mask(self):
        return self.mask
    def get_obj_masked(self):
        return self.obj_masked
    def get_obj_class(self):
        return self.obj_class
    def get_obj_id(self):
        return self.obj_id
    

class CopyPasteBank():
    def __init__(self,buffer_size=200,edge_margin=0.05):
        super().__init__()
        self.size = buffer_size
        self.edge_margin = edge_margin #for each side
        self.bank = deque()
        self.count_occlusion = 0
        self.count_pastes = 0
        #for pasting strategy
        self.elements = [2,7,12,17,22,27] #acording to armbench
        probabilities = [0.448,0.134,0.119,0.118,0.123,0.058]#[1/0.37, 1/0.21, 1/0.14,1/0.1,1/0.07,1/0.12]
        self.probabilities = probabilities/np.array(probabilities).sum()
        
    def append(self,imgs,instances):
        """
        add one element to the bank
        note that mask is singular
        """
        for i,img in enumerate(imgs):
            for j in range(len(instances[i].gt_classes)): 
                if len(instances[i].gt_classes) < 2: #empty tote?
                    continue
                obj_class = instances[i].gt_classes[j]
                if int(obj_class) == 0: #skip the tote
                    continue
                obj_id = instances[i].gt_ids[j]
                
                obj_mask = instances[i].gt_masks.tensor[j]
                obj_masked = img * obj_mask

                obj_cords = torch.nonzero(obj_mask)
                if obj_mask.sum()==0:
                    continue
                x1,y1 = obj_cords.min(0)[0]
                x2,y2 = obj_cords.max(0)[0]
                if x2 > (img.shape[1] - 20) or y2 > (img.shape[2] - 20):
                    continue
                obj_masked = obj_masked[:,x1:x2,y1:y2]
                mask = obj_mask[x1:x2,y1:y2]
                w,h = x2-x1,y2-y1

                #img[:,x+w,y:y+h] = obj_masked * mask + img[:,x+w,y:h] * ~mask
                
                #todo: the box and mask coordinates does not help.. we only need to size
                #when pasting we need to generate boxes etc...    
                if len(self.bank) == self.size:
                    self.bank.popleft()
                self.bank.append(image_obj(obj_masked.to('cpu'),mask.to('cpu'),obj_class.to('cpu'),obj_id,(w,h)))

    def paste(self,img,gt_instance,n_objects=None):
        #if gt_instance._fields['gt_masks'].tensor.shape[1:] != img.shape[1:]:
        #    return gt_instance
        if not n_objects:
            if np.random.rand() > 1.1:
                n_objects = self.num_to_paste(len(gt_instance.gt_classes))
            else:
                n_objects = np.random.randint(5)
            #n_objects = 1  
        if n_objects > len(self.bank): #empty bank, don't paste
            return gt_instance, 0
        #sample n objects from bank
        sample_indices = sample(range(len(self.bank)), n_objects)
        objects = [self.bank[index] for index in sample_indices]
        
        for object in objects:
            #create the image with object
            x2,y2 = object.get_shape()       
            start_x,start_y=np.random.beta(.5,.5,2)
            valid_w = img.shape[1]*(1-self.edge_margin) - x2
            valid_h = img.shape[2]*(1-self.edge_margin) - y2
            x1 = int(self.edge_margin * img.shape[1] + start_x*valid_w)
            y1 = int(self.edge_margin * img.shape[2] + start_y*valid_h)
            
            if x1+x2 >= img.shape[1] or y1+y2 >= img.shape[2]:
                #object does not fit with margin, TODO: try to resize object
                continue
            
            mask = object.get_mask().to(img.device) 
            obj_masked = object.get_obj_masked().to(img.device) #the object itself
            device = gt_instance.gt_boxes.device
            
            obj_mask = self.generate_object_mask(object,img.shape[1:],(x1,y1,x2,y2),device).unsqueeze(0)
            for current_mask in gt_instance.gt_masks.tensor:
                mask_size = current_mask.sum()
                current_mask*= ~obj_mask[0] #correct masks
                #do not copy if ocluda 80% of object

                if current_mask.sum() < mask_size*.1:
                    #self.count_occlusion+=1
                    #print(f'# occlusions over 80%: {self.count_occlusion}')
                    continue
                #else: 
                    #self.count_pastes+=1
                    #if self.count_pastes % 1000 == 0:
                    #    print(f'# pastes: {self.count_pastes}')
            #add to instance
            gt_instance.gt_masks.tensor = torch.cat((gt_instance.gt_masks.tensor,obj_mask))
            
            # add to image, no blending
            img[:,x1:x1+x2,y1:y1+y2] = obj_masked * mask + img[:,x1:x1+x2,y1:y1+y2] * ~mask


            #box assignment:
            obj_box = torch.tensor((x1,y1,x1+x2,y1+y2)).unsqueeze(0).to(device)
            assert (obj_box[:, 2:] >= obj_box[ :,:2]).all()
            gt_instance.gt_boxes.tensor = torch.cat((gt_instance.gt_boxes.tensor,obj_box))
            obj_class = object.get_obj_class().unsqueeze(0).to(device)

            if len(gt_instance.gt_classes.shape)==2:
                gt_instance.gt_classes = gt_instance.gt_classes.squeeze(0)    
            
            gt_instance.gt_classes = torch.cat((gt_instance.gt_classes,obj_class))
            
            
            obj_id = object.get_obj_id().unsqueeze(0).to(device)
            gt_instance.gt_ids = torch.cat((gt_instance.gt_ids,obj_id))
        return gt_instance, n_objects
    

    def num_to_paste(self,curr_obj_num=1):
        total_obj_sampled = np.random.choice(self.elements, 1, p=self.probabilities)[0]
        if curr_obj_num >= total_obj_sampled:
            return 0
        else:
            return total_obj_sampled - curr_obj_num
        
    def generate_object_mask(self,image_obj,image_size,xyxy_box,device):
        x1,y1,x2,y2 = xyxy_box
        new_mask = torch.zeros(image_size).type(torch.bool).to(device)
        obj_mask = image_obj.get_mask().to(device)
        new_mask[x1:x1+x2,y1:y1+y2] += obj_mask
        return new_mask

    def __len__(self):
        return len(self.bank)

def tv_save_image(image,name='masked.png'):
    img = (image - image.min())/(image.max()-image.min())
    save_image(img, name) 

def save_imgs(images_w,images_s):
    for i,_ in enumerate(images_w.tensor):
        image = images_w.tensor[i]
        img = (image - image.min())/(image.max()-image.min())
        save_image(img, f'w_{i}.png')
        image = images_s.tensor[i]
        img = (image - image.min())/(image.max()-image.min())
        save_image(img, f's_{i}.png')


def draw_masks(image, masks, alpha=0.6):
    update = alpha
    momentum = 1.0 - alpha
    for mask in masks:
        color = np.random.random(3) * 255
        _mask = mask['segmentation'] > 0
        image[_mask, :] = momentum * image[_mask, :] + update * color

    return image