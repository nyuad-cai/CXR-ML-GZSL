import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
# import 
import glob


class NIHChestXray(Dataset):

    def __init__ (self, args, pathDatasetFile, transform, classes_to_load='seen', exclude_all=True):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        self.num_classes = args.num_classes

        self._data_path = args.data_root
        self.args = args
        
        self.split_path = pathDatasetFile
        self.CLASSES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        



        self.unseen_classes = ['Edema', 'Pneumonia', 'Emphysema', 'Fibrosis']

        self.seen_classes = [ 'Atelectasis', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                'Pneumothorax', 'Consolidation', 'Cardiomegaly', 'Pleural_Thickening', 'Hernia']

        self._class_ids = {v: i for i, v in enumerate(self.CLASSES) if v != 'No Finding'}

        self.seen_class_ids = [self._class_ids[label] for label in self.seen_classes]
        self.unseen_class_ids = [self._class_ids[label] for label in self.unseen_classes]
        
        
        self.classes_to_load = classes_to_load
        self.exclude_all = exclude_all
        self._construct_index()
        
    def _construct_index(self):
        # Compile the split data path
        max_labels = 0
        paths = glob.glob(f'{self._data_path}/**/images/*.png')
        self.names_to_path = {path.split('/')[-1]: path for path in paths}
        data_entry_file = 'Data_Entry_2017.csv'
        # split_path = os.path.join(self._data_path, self._split)
        print(f'data partition path: {self.split_path}')
        with open(self.split_path, 'r') as f: file_names = f.readlines()


        split_file_names = np.array([file_name.strip().split(' ')[0].split('/')[-1] for file_name in file_names])
        df = pd.read_csv(f'{self._data_path}/{data_entry_file}')
        image_index = df.iloc[:, 0].values

        _, split_index, _ = np.intersect1d(image_index, split_file_names, return_indices=True)
        


        labels = df.iloc[:, 1].values
        labels = np.array(labels)[split_index]

        labels = [label.split('|') for label in labels]

        image_index = image_index[split_index]


        # remove No Finding

        # Construct the image db
        self._imdb = []
        self.class_ids_loaded = []
        for index in range(len(split_index)):
            if len(labels[index]) == 1 and labels[index][0] == 'No Finding':
                continue
            if self._should_load_image(labels[index]) is False:
                continue
            class_ids = [self._class_ids[label] for label in labels[index]]
            self.class_ids_loaded +=class_ids
            self._imdb.append({
                'im_path': self.names_to_path[image_index[index]],
                'labels': class_ids,
            })
            max_labels = max(max_labels, len(class_ids))
        
        # import pdb; pdb.set_trace()
        self.class_ids_loaded = np.unique(np.array(self.class_ids_loaded))
        print(f'Number of images: {len(self._imdb)}')
        print(f'Number of max labels per image: {max_labels}')
        print(f'Number of classes: {len(self.class_ids_loaded)}')


    def _should_load_image(self, labels):


        selected_class_labels = self.CLASSES
        if self.classes_to_load == 'seen':
            selected_class_labels = self.seen_classes
        elif self.classes_to_load == 'unseen':
            selected_class_labels = self.unseen_classes
        elif self.classes_to_load == 'all':
            return True
        
        count = 0
        for label in labels:
            if label in selected_class_labels:
                count+=1
           
        if count == len(labels):
            # all labels from selected sub set
            return True
        elif count == 0:
            # none label in selected sub set
            return False
        else:
            # some labels in selected sub set
            if self.exclude_all is True:
                return False
            else:
                return True
            

    
    def __getitem__(self, index):
        
        imagePath = self._imdb[index]['im_path']
        
        imageData = Image.open(imagePath).convert('RGB')

        labels = torch.tensor(self._imdb[index]['labels'])

        labels = labels.unsqueeze(0)
        imageLabel = torch.zeros(labels.size(0), self.num_classes).scatter_(1, labels, 1.).squeeze()
        
        img = self.transform(imageData)
        return img, imageLabel

    
    def __len__(self):
        
        return len(self._imdb)
    
    