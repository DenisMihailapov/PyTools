import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms, io
import torchvision.transforms.functional  as TF


def train_val_split(DIR_PATH, TEST_SIZE = 0.2, SEED = 42):
    
    generator=torch.Generator().manual_seed(SEED)


    total_size = 0; test_size = 0
    dir_list = []; out_list=[] 

    print('load data from:')
    for c in os.listdir(DIR_PATH):#for classes
        print(DIR_PATH + '/' + c + '/')
        
        dir_list = os.listdir(DIR_PATH + '/' + c +'/')
        total_size = len(dir_list)
        test_size = int(TEST_SIZE*total_size)

        out_list.append(random_split(dir_list, [total_size-test_size, test_size], generator=generator))
        
    return out_list


def get_train_val_dataset_split(DIR_PATH, BATCH_SIZE,  TEST_SIZE = 0.2, SEED = 42):

    train_val_pathes_list = train_val_split(DIR_PATH, TEST_SIZE, SEED)

    train, valid = [], []

    for i in range(len(train_val_pathes_list)):#for classes
        train.append(train_val_pathes_list[i][0])
        valid.append(train_val_pathes_list[i][1])



    train_loader = DataLoader(ImageClassifDataset(train,  DIR_PATH, True),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ImageClassifDataset(valid,  DIR_PATH, False), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


class ImageClassifDataset(Dataset):
    def __init__(self, list_of_classes_pathes, data_path, train_of_test=True, img_size=(224, 224)):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.data_path = data_path

        self.classes = os.listdir(data_path)
        self.classes.sort(reverse=True)
        
        self.list_of_classes_pathes = list_of_classes_pathes

        self.samples_list = []

        for i, L in enumerate(list_of_classes_pathes):
           self.samples_list.extend([(path, i) for path in L])

        resize = transforms.Resize(img_size)
        #noise_tr = resize(0.05*torch.randn((3, 6, 6), device=self.device))   
        #noise_te = resize(0.010*torch.randn((3, 6, 6),   device=self.device))

        if train_of_test:
            self.transforms = transforms.Compose([
                                  resize,
                                  transforms.GaussianBlur(5, (0.1, 3.0)),
                                  transforms.Normalize(0, 255),

                                  #transforms.Lambda(lambda x: x + noise_tr),
                                  transforms.RandomRotation(30)

            ])
        else:
            self.transforms = transforms.Compose([
                          resize,
                          transforms.GaussianBlur(5, (0.1, 1.0)),
                          transforms.Normalize(0, 255),
                          #transforms.Lambda(lambda x: x + noise_te)

            ])


    def __getitem__(self, idx):
        s = self.samples_list[idx]

             
        img = io.read_image(self.data_path + '/' + \
                            self.classes[s[1]]+ '/' + \
                            s[0]).float().to(self.device)
      
        label = torch.tensor([s[1]], dtype=torch.float, device=self.device)
  
        return self.transforms(img), label

    def __len__(self):
        return len(self.samples_list)
