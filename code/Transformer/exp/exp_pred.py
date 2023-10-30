from cgi import test
from torch.utils.data.dataset import Dataset
from data.stock_data_handle import Stock_Data,DatasetStock,DatasetStock_PRED
from exp.exp_basic import Exp_Basic
from models.transformer import Transformer_base as Transformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, ranking_loss
import utils.tools as utils
import utils.metrics_object as metrics_object

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pdb

import os
import time

dataset_dict = {
    'stock': DatasetStock_PRED,
}

class Exp_pred(Exp_Basic):
    def __init__(self, args, data_all, id):
        super(Exp_pred, self).__init__(args)
        log_dir = os.path.join('log', 'pred_'+args.project_name+'_'+str(args.rank_alpha)+'_'+id)
        print(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.data_all = data_all
    
    def _build_model(self):
        model_dict = {
            'Transformer':Transformer,
        }

        if self.args.model=='Transformer':
            model = model_dict[self.args.model](
                # self.args
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.activation
                # self.device
            )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model.float()

    def _get_data(self, flag):
        args = self.args

        if flag == 'train':
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size
        else:
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
      
        dataset = dataset_dict[self.args.data_type](self.data_all, type=flag, pred_type=self.args.pred_type)
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, metric_builders, stage='test'):
        self.model.eval()
        total_loss = []
        metric_objs = [builder(stage) for builder in metric_builders]

        for i, (batch_x1, batch_x2, batch_y) in enumerate(vali_loader):
            bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
            batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
            batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            _, _, output = self.model(batch_x1, batch_x2)

            output = output.reshape(bs,stock_num)
            loss = criterion(output, batch_y) + self.args.rank_alpha * ranking_loss(output, batch_y)

            total_loss.append(loss.item())

            with torch.no_grad():
                for metric in metric_objs:
                    metric.update(output, batch_y)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, metric_objs
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'valid')
        test_data, test_loader = self._get_data(flag = 'test')

        metrics_builders = [
        metrics_object.MIRRTop1,
    ]

        path = os.path.join('./checkpoints/',setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        metric_objs = [builder('train') for builder in metrics_builders]

        valid_loss_global = np.inf
        best_model_index = -1

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            for i, (batch_x1, batch_x2, batch_y) in enumerate(train_loader):
                iter_count += 1
                # pdb.set_trace()
                bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
                batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
                batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                _,_, output = self.model(batch_x1, batch_x2)
            
                output = output.reshape(bs,stock_num)
        
                loss = criterion(output, batch_y) + self.args.rank_alpha * ranking_loss(output, batch_y)
                train_loss.append(loss.item())

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                with torch.no_grad():
                    for metric in metric_objs:
                        metric.update(output, batch_y)


            train_loss = np.average(train_loss)
            valid_loss, valid_metrics = self.vali(vali_data, vali_loader, criterion, metrics_builders, stage='valid')
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion, metrics_builders, stage='test')

            self.writer.add_scalar('Train/loss', train_loss, epoch)
            self.writer.add_scalar('Valid/loss', valid_loss, epoch)
            self.writer.add_scalar('Test/loss', test_loss, epoch)

            # pdb.set_trace()

            all_logs = {
                metric.name: metric.value for metric in metric_objs + valid_metrics + test_metrics
            }
            for name, value in all_logs.items():
                self.writer.add_scalar(name, value.mean(), global_step=epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Valid Loss: {3:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))
            
            torch.save(self.model.state_dict(), path+'/'+'checkpoint_{0}.pth'.format(epoch+1))

            if valid_loss.item() < valid_loss_global:
                best_model_index = epoch+1

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint_{0}.pth'.format(best_model_index)
        self.model.load_state_dict(torch.load(best_model_path))
        print('best model index: ', best_model_index)
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        outputs = []
        real = []
        
        self.model.eval()

        metrics_builders = [
        metrics_object.MIRRTop1,
        metrics_object.RankIC
    ]
        
        metric_objs = [builder('test') for builder in metrics_builders]
        
        for i, (batch_x1, batch_x2, batch_y) in enumerate(test_loader):
            bs, stock_num = batch_x1.shape[0], batch_x2.shape[1]
            batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
            batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            _,_, output = self.model(batch_x1, batch_x2)

            output = output.reshape(bs,stock_num)

            with torch.no_grad():
                for metric in metric_objs:
                    metric.update(output, batch_y)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        all_logs = {
                metric.name: metric.value for metric in metric_objs
            }
        for name, value in all_logs.items():
            print(name, value.mean())

        return 