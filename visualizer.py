import torch
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    
    def __init__(self, model, dataset, device='cuda'):
        self._model = model
        self._dataset = dataset
        self._device = device


    def _show(self, imgs):
        if not isinstance(imgs, list): 
            imgs = [imgs]
        l = len(imgs)    
        fix, axs = plt.subplots(ncols=l, squeeze=False, figsize=(5*l,5*l))
        for i, img in enumerate(imgs):
            img = img.detach()
            img = TF.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


    def show_data(self, indexes = [], with_model=False):
        
        if len(indexes) == 0: 
            indexes = list(np.random.randint(0, len(self._dataset), 5))
            print('Indexes:', indexes)
         
        num_examples = len(indexes); 
        
        img, mask = self._dataset[indexes[0]].values(); num_examples-=1; 

        if with_model: 
            self._model.eval()
            pred = self._model(img.unsqueeze(0).to(self._device))[0][0]
            
            print('    ','Image','\t\t\t\t','True Mask','\t\t\t','Predict')
            print('    ',img.shape, '\t\t', mask.shape, '\t', pred.shape)
            print('    ',img.type(),'\t\t', mask.type(),'\t', pred.type())

            self._show([img, mask, pred]) 
        else:
            print('    ','Image','\t\t\t\t','Mask')
            print('    ',img.shape,'\t\t', mask.shape)
            self._show([img, mask]) 
   

        for i in range(1, num_examples):
            img, mask = self._dataset[indexes[i]].values()
            if with_model: 
                pred = self._model(img.unsqueeze(0).to(self._device))[0][0]
                self._show([img, mask, pred])
            else: self._show([img, mask])        
