import numpy as np
import time
from datetime import datetime, timedelta
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import accuracy_score
from zsl_models import ZSLNet
from dataset import NIHChestXray
from torch.nn.functional import kl_div, softmax, log_softmax
from numpy import dot
from numpy.linalg import norm
from plots import plot_array
# #-------------------------------------------------------------------------------- 

    
class ChexnetTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.textual_embeddings = np.load(args.textual_embeddings)

        self.model = ZSLNet(self.args, self.textual_embeddings, self.device).to(self.device)
        self.optimizer = optim.Adam (self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # self.optimizer  = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.0001)
        # self.scheduler = self.step_lr
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')
        

        self.loss = torch.nn.BCELoss(size_average=True)
        self.auroc_min_loss = 0.0

        self.start_epoch = 1
        self.lossMIN = float('inf')
        self.max_auroc_mean = float('-inf')
        self.best_epoch = 1
        
        self.val_losses = []
        print(self.model)
        print(self.optimizer)
        print(self.scheduler)
        print(self.loss)
        print(f'\n\nloaded imagenet weights {self.args.pretrained}\n\n\n')
        self.resume_from()
        self.load_from()
        self.init_dataset()
        self.steps = [int(step) for step in self.args.steps.split(',')]
        self.time_start = time.time()
        self.time_end = time.time()
        self.should_test = False
        self.model.class_ids_loaded = self.train_dl.dataset.class_ids_loaded


    def __call__(self):
        self.train()
    
    def step_lr(self, epoch):
        step = self.steps[0]
        for index, s in enumerate(self.steps):
            if epoch < s:
                break
            else:
                step = s

        lr = self.args.lr * (0.1 ** (epoch // step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_from(self):
        if self.args.load_from is not None:
            checkpoint = torch.load(self.args.load_from)
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f'loaded checkpoint from {self.args.load_from}')

    def resume_from(self):
        if self.args.resume_from is not None:
            checkpoint = torch.load(self.args.resume_from)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.lossMIN = checkpoint['lossMIN']
            self.max_auroc_mean = checkpoint['max_auroc_mean']
            print(f'resuming training from epoch {self.start_epoch}')

    def save_checkpoint(self, prefix='best'):
        path = f'{self.args.save_dir}/{prefix}_checkpoint.pth.tar'
        torch.save(
            {
            'epoch': self.epoch, 
            'state_dict': self.model.state_dict(), 
            'max_auroc_mean': self.max_auroc_mean, 
            'optimizer' : self.optimizer.state_dict(),
            'lossMIN' : self.lossMIN
            }, path)
        print(f"saving {prefix} checkpoint")
    def init_dataset(self):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        train_transforms = []
        train_transforms.append(transforms.RandomResizedCrop(self.args.crop))
        train_transforms.append(transforms.RandomHorizontalFlip())
        train_transforms.append(transforms.ToTensor())
        train_transforms.append(normalize)      

        datasetTrain = NIHChestXray(self.args, self.args.train_file, transform=transforms.Compose(train_transforms))
        datasetVal =   NIHChestXray(self.args, self.args.val_file, transform=transforms.Compose(train_transforms))

        self.train_dl = DataLoader(dataset=datasetTrain, batch_size=self.args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        self.val_dl = DataLoader(dataset=datasetVal, batch_size=self.args.batch_size*10, shuffle=False, num_workers=4, pin_memory=True)


        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        
        datasetTest = NIHChestXray(self.args, self.args.test_file, transform=transforms.Compose(test_transforms), classes_to_load='all')
        self.test_dl = DataLoader(dataset=datasetTest, batch_size=self.args.batch_size*3, num_workers=8, shuffle=False, pin_memory=True)
        print(datasetTest.CLASSES)
        

    def train (self):
        
        for self.epoch in range (self.start_epoch, self.args.epochs):

            self.epochTrain()
            lossVal, val_ind_auroc = self.epochVal()
            val_ind_auroc = np.array(val_ind_auroc)


            aurocMean = val_ind_auroc.mean()
            self.save_checkpoint(prefix=f'last_epoch')  
            self.should_test = False

            if aurocMean > self.max_auroc_mean:
                self.max_auroc_mean = aurocMean
                self.save_checkpoint(prefix='best_auroc')
                self.best_epoch = self.epoch
                self.should_test = True
            if lossVal < self.lossMIN:
                self.lossMIN = lossVal
                self.auroc_min_loss = aurocMean
                self.save_checkpoint(prefix='min_loss')
                self.should_test = True

            self.print_auroc(val_ind_auroc, self.val_dl.dataset.class_ids_loaded, prefix='val')
            if self.should_test is True:
                test_ind_auroc = self.test()
                test_ind_auroc = np.array(test_ind_auroc)
               
                self.write_results(val_ind_auroc, self.val_dl.dataset.class_ids_loaded, prefix=f'\n\nepoch {self.epoch}\nval', mode='a')

                self.write_results(test_ind_auroc[self.test_dl.dataset.seen_class_ids], self.test_dl.dataset.seen_class_ids, prefix='\ntest_seen', mode='a')
                self.write_results(test_ind_auroc[self.test_dl.dataset.unseen_class_ids], self.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen', mode='a')

                self.print_auroc(test_ind_auroc[self.test_dl.dataset.seen_class_ids], self.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
                self.print_auroc(test_ind_auroc[self.test_dl.dataset.unseen_class_ids], self.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')
            
            plot_array(self.val_losses, f'{self.args.save_dir}/val_loss')
            print(f'best epoch {self.best_epoch} best auroc {self.max_auroc_mean} loss {lossVal:.6f} auroc at min loss {self.auroc_min_loss:0.4f}')
            
            self.scheduler.step(lossVal)

                     
    #-------------------------------------------------------------------------------- 
    def get_eta(self, epoch, iter):
        self.time_end = time.time()
        delta = self.time_end - self.time_start
        delta = delta * (len(self.train_dl) * (self.args.epochs - epoch) - iter)
        sec = timedelta(seconds=int(delta))
        d = (datetime(1,1,1) + sec)
        eta = f"{d.day-1} Days {d.hour}:{d.minute}:{d.second}"
        self.time_start = time.time()

        return eta



    

    def epochTrain(self):
        self.model.train()
        epoch_loss = 0
        for batchID, (inputs, target) in enumerate (self.train_dl):

            target = target.to(self.device)
            inputs = inputs.to(self.device)
            output, loss = self.model(inputs, target, self.epoch)
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            eta = self.get_eta(self.epoch, batchID)
            epoch_loss +=loss.item()
            if batchID % 10 == 9:
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] eta: {eta:<20} [{batchID:04}/{len(self.train_dl)}] lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/batchID:0.5f}")

            
            
    #-------------------------------------------------------------------------------- 
        
    def epochVal (self):
        
        self.model.eval()
        
        lossVal = 0
        
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        for i, (inputs, target) in enumerate (tqdm(self.val_dl)):
            with torch.no_grad():
            
                target = target.to(self.device)
                inputs = inputs.to(self.device)
                varInput = torch.autograd.Variable(inputs)
                varTarget = torch.autograd.Variable(target)    
                varOutput, losstensor = self.model(varInput, varTarget)

                outPRED = torch.cat((outPRED, varOutput), 0)
                outGT = torch.cat((outGT, target), 0)

                lossVal+=losstensor.item()
                del varOutput, varTarget, varInput, target, inputs
        lossVal = lossVal / len(self.val_dl)
        
        aurocIndividual = self.computeAUROC(outGT, outPRED, self.val_dl.dataset.class_ids_loaded)
        self.val_losses.append(lossVal)

        return lossVal, aurocIndividual
    
    
   
    def test(self):
        cudnn.benchmark = True
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        self.model.eval()
        
        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):
            with torch.no_grad():
                target = target.to(self.device)
                outGT = torch.cat((outGT, target), 0)
                
                bs, n_crops, c, h, w = inputs.size()
                
                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))
                
                out, _ = self.model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
                
                outPRED = torch.cat((outPRED, outMean.data), 0)
                


        aurocIndividual = self.computeAUROC(outGT, outPRED, self.test_dl.dataset.class_ids_loaded)
        
        aurocMean = np.array(aurocIndividual).mean()
        
        return aurocIndividual
    
    def computeAUROC (self, dataGT, dataPRED, class_ids):
        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in class_ids:
            # np.unique(np.array(self.val_dl.dataset.class_ids_loaded)):  
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        return outAUROC

    def write_results(self, aurocIndividual, class_ids, prefix='val', mode='a'):

        with open(f"{self.args.save_dir}/results.txt", mode) as results_file:

            aurocMean = aurocIndividual.mean()

            results_file.write(f'{prefix} AUROC mean {aurocMean:0.4f}\n')
            for i, class_id in enumerate(class_ids):  
                results_file.write(f'{self.val_dl.dataset.CLASSES[class_id]} {aurocIndividual[i]:0.4f}\n')
        
    def print_auroc(self, aurocIndividual, class_ids, prefix='val'):
        aurocMean = aurocIndividual.mean()

        print (f'{prefix} AUROC mean {aurocMean:0.4f}')
        
        for i, class_id in enumerate(class_ids):  
            print (f'{self.val_dl.dataset.CLASSES[class_id]} {aurocIndividual[i]:0.4f}')
        



