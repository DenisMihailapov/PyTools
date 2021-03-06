#This trainer was based on trainer from https://github.com/Emorh/SHIFT_SegmentationProject
import os, torch, wandb

from tqdm.notebook import tqdm
from collections import defaultdict


# TODO: do metrics lists ad visualize it (matby in notebook) 

class Trainer:
    CHECKPOINTS_PATH = 'checkpoints'
    
    def __init__(self, model, criterion, metric, config, device='cuda', path = '.', save_file_name = 'weights.pth', project_name='segmentation project'):
        
        self._model     = model
        self._criterion = criterion
        self._metrics   = metric
        self._device    = device
        self._save_file_name = save_file_name
        self._path = path
        self._config = config
        self._project_name = project_name
         
        
        self._model.to(self._device)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config['lr'])
        
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            'max',
            factor=self._config['lr_reduce_rate'],
            patience=self._config['patience'],
            verbose=True
        )
        
        wandb.login()

        self._best_metric = float('-inf')
        if not os.path.exists(Trainer.CHECKPOINTS_PATH):
            os.makedirs(Trainer.CHECKPOINTS_PATH)
    
    def predict(self, image):
        return self._model(images)
    
    def print_config(self):
        print("Trainer configuration")
        for key, value in self._config.items():
            print(f"{key}: {value}")
        print('-'*15)    

    def update_param(self, model=None, criterion=None, metric=None, config=None, device=None, path = None, save_file_name = None, project_name= None ):
        if not model is None: self._model = model
        elif not criterion is None: self._criterion = criterion
        elif not metric is None: self._metric = metric
        elif not config is None: self._config = config
        elif not device is None: self._device = device
        elif not path is None: self._path = path
        elif not save_file_name is None: self._save_file_name = save_file_name
        elif not project_name is None: self._project_name = project_name
        else: print('There was no update')
    
    def load_weights(self, file_name = 'weights.pth'):
        self._save_file_name = file_name
        self._model.load_state_dict(torch.load(self._get_full_path(file_name)))
    
    def fit(self, train_loader, val_loader):
        self.print_config()
        with wandb.init(project=self._project_name , config=self._config):
            wandb.watch(self._model, self._criterion, log='all', log_freq=10)
            self._fit(train_loader, val_loader)
        
    def _fit(self, train_loader, val_loader):

        passed_epochs_without_upgrades = 0

        for epoch in range(1, self._config['epochs']+1):
            print(f'Epoch {epoch}')
            if passed_epochs_without_upgrades > self._config['early_stopping']:
                print('Early Stopping!')
                return 
            
            ##Train##
            print('Train')
            self._model.train()
            train_metrics = self._run_epoch(epoch, train_loader, is_training=True)

            metrics_str = []
            for name, value in train_metrics.items():
                metrics_str.append(f'{name}: {float(value):.5f}')
            metrics_str = ' '.join(metrics_str)
            print('train metrics: ' + metrics_str); print()
            ###########  


            #Validation  
            print('Validation')
            self._model.eval()
            val_metrics = self._run_epoch(epoch, val_loader, is_training=False)

            self._scheduler.step(val_metrics['dice'])

            metrics_str = []
            for name, value in val_metrics.items():
                metrics_str.append(f'{name}: {float(value):.5f}')
            metrics_str = ' '.join(metrics_str)
            print('val metrics: ' + metrics_str); print()
            ###########
            
            

            if self._best_metric < val_metrics['dice']:
                passed_epochs_without_upgrades = 0

                print(f"Save {self._best_metric} -> {round(val_metrics['dice'], 4) } (dice)")
                torch.save(self._model.state_dict(), self._get_full_path(self._save_file_name))
                self._best_metric = round(val_metrics['dice'], 4)
           
            print('-'*10,'\n')
            passed_epochs_without_upgrades += 1
    
    def _get_full_path(self, file_name):
        return os.path.join(self._path, Trainer.CHECKPOINTS_PATH, file_name)
    
    def _run_epoch(self, epoch, loader, is_training):

        avg_metrics = defaultdict(float)
        
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            
            batch_metrics = self._step(data, is_training)
            for name, val in batch_metrics.items():
                avg_metrics[name] += val
        
        if not is_training:
            for name, val in avg_metrics.items():
                wandb.log({name: val / len(loader)})
                
        return {name: value / len(loader) for name, value in avg_metrics.items()}
    
    def _step(self, data, is_training=True):
        metrics_values = {}
        images = data['image'].detach().to(self._device).detach()
        y_true = data['mask'].detach() .to(self._device).detach()
        
        if is_training:
            self._optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            y_pred = self._model(images)
            loss   = self._criterion(y_pred, y_true)
            
            for name, func in self._metrics:
                value = func(y_true=y_true, y_pred=torch.sigmoid(y_pred))
                metrics_values[name] = value.item()

            if is_training:
                loss.backward()
                self._optimizer.step()
        
        metrics_values['loss'] = loss.item()
        return metrics_values
        
        
        
        
        
        
