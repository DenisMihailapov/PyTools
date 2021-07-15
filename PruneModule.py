import os
print('install torch_pruning...')
print('this lib was taken from https://github.com/VainF/Torch-Pruning')
os.execute('!pip install torch_pruning')
import torch_pruning as tp



from tqdm.notebook import tqdm
import numpy as np 
from collections import Counter

import torch
import torch.nn as nn 
import torch.nn.functional as F


__all__strategy__ = ['l1','l2','random']

class PruneModule:
      ########### Privet block ############  
  def __full_model_path(self):
          return self.root_path+'/chekpoints/'+self.model_name+'.pt' 

  def __get_size_of_model(self):
      torch.save(self.model.state_dict(), "./temp.p")
      s = os.path.getsize("./temp.p")/1e6
      os.remove('./temp.p')
      return s

  def __get_count_params_of_model(self):
      return sum([np.prod(p.size()) for p in self.model.parameters()])/1e6
       

  def __evalute(self):
      print('\nTest model') 

      correct = 0;total = 0
    
      self.model.to(self.DEVICE)
      self.model.eval()

      with torch.no_grad():
          for img, target in tqdm(self.test_loader):
              out = self.model(img.to(self.DEVICE))
            
              pred = out.max(1)[1].detach().cpu().numpy()
              target = target.cpu().numpy()
              correct += (pred==target).sum()
              total += len(target)

      return round(correct / total, 3)

  def __getListsofType(self):
    L = []
    for module in self.model.modules(): 
      par = sum(p.numel() for p in module.parameters())
      L.append((module, par))

    LTypes = [(l[1],type(l[0])) for l in L]
    LSTypes = list(set(LTypes))


    LType = [type(l[0]) for l in L]
    LSType = list(set(LType))

    return LTypes, LSTypes, LType, LSType   


  def __init__(self, model, train_loader, test_loader, model_name = 'model', DEVICE = 'auto', 
                     input_example = torch.randn(1, 3, 28, 28), root_path='.',
                     strategy = 'l1', conv_amount = 0.1, linear_amount = 0.1):

    self.model      = model
    self.model_name = model_name

    self.train_loader = train_loader
    self.test_loader  = test_loader
    self.root_path    = root_path

    self.conv_amount    = conv_amount
    self.linear_amount  = linear_amount

    if DEVICE == 'auto': 
      self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: self.DEVICE = DEVICE


    self.DG = tp.DependencyGraph().build_dependency(self.model, input_example)

    self.set_pruning_strategy(strategy)

  ##############################################################################

  def set_pruning_strategy(self, strategy):
    if strategy == 'l1':
            self.strategy = tp.strategy.L1Strategy()
    elif strategy == 'l2':
        self.strategy = tp.strategy.L2Strategy()    
    elif strategy == 'random':
        self.strategy = tp.strategy.L2Strategy()   
    else:
        print('Unknown strategy', strategy)
        print('Strategy is L1 now')
        self.strategy = tp.strategy.L1Strategy() 

  def print_size_of_model(self):
        print('Size of the model(MB):', self.__get_size_of_model())

  def print_count_params_of_model(self):
        print("Number of Parameters: %.1fM"%(self.__get_count_params_of_model()))     

  def print_accuracy(self):
       print('Accuracy: ', self.__evalute())      


  def train_test_loop(self, opt = 'SGD', EPOCH = 1, lr = 0.01, momentum=0.9, load_first=False):
      print('Train model')

      if os.path.exists(self.__full_model_path()) and load_first:
          self.model = torch.load(self.__full_model_path())

    
      if opt == 'SGD':
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
      elif opt == 'Adam':
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
      else:
        print('Unknown optim', opt)
        return


      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 70, 0.1)
      self.model.to(self.DEVICE)

      best_acc = self.__evalute()
      print('\nAccuracy now: ', best_acc)
    
      for epoch in range(1,EPOCH+1):
          print("\nEpoch %d/%d"%(epoch, EPOCH))
          self.model.train()
          for img, target in tqdm(self.train_loader):
              img, target = img.to(self.DEVICE), target.to(self.DEVICE)

              optimizer.zero_grad()

              out = self.model(img)

              loss = F.cross_entropy(out, target)
              loss.backward()

              optimizer.step()



          self.model.eval()
          acc = self.__evalute()
          print("Accuracy: ", acc)

          if best_acc < acc:
            print('Save', best_acc,'-->', acc)
            torch.save(self.model, self.__full_model_path())
            best_acc = acc

          scheduler.step()
          print()

      self.model = torch.load(self.__full_model_path())
      print("\nBest Acc=%.4f"%(best_acc))

  def modules_with_patametrs(self, print_info = False):
    LTypes,_,LType,_ = self.__getListsofType()
    LTypesWithoutParam = [pair[1] for pair in LTypes if pair[0] !=0 ]

    Dtemp = Counter(LType)
    D = dict()

    for k, v in Dtemp.items():
      for key in LTypesWithoutParam:
        if key == k: D.update({k:v})
      
    if print_info: 
        print(D) #Modules with parametrs
    else:
        return D

  def print_model_size_accuracy_params(self):
    self.print_accuracy()
    self.print_size_of_model()
    self.print_count_params_of_model()

  def prune_conv(self, module, amount):

    prune_index = self.strategy(module.weight, amount=amount)
    plan = self.DG.get_pruning_plan(module, tp.prune_conv, prune_index)
    plan.exec()

  def prune_model(self):
    self.model.cpu()
    
    for module in self.model.modules():
        if isinstance(module, nn.Conv2d):

            plan = self.DG.get_pruning_plan(module, tp.prune_conv, 
                                            self.strategy(module.weight, 
                                                          amount=self.conv_amount))
            plan.exec()

        elif isinstance(module, nn.Linear):
    
            plan = self.DG.get_pruning_plan(module, tp.prune_linear, 
                                            self.strategy(module.weight, 
                                                          amount=self.linear_amount))
            plan.exec()    
            
            
    return self.model

  def run_main_cycle(self, border_size = 2.5, start_lr = 0.005):
      
      lr = start_lr
      i = 1

      self.__print_accuracy()
      size = self.__get_size_of_model()
      print('Size of the model(MB):', size)

      while(size >= border_size): #MB
        print("=======Cycle %d======="%(i)); i+=1
        print("\nPrune...")
        self.prune_model()

        size = self.__get_size_of_model()
        print('\nAccuracy after prune', self.__evalute())
        print('Size of the model(MB):', size)
        self.print_count_params_of_model()


        torch.save(self.model, self.__full_model_path())
  
        print('\n\n')
        self.train_test_loop(EPOCH=3, lr=lr)
        lr /= 4

        print()
        print("Result:")
        self.print_model_size_accuracy_params()
