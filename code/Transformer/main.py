import argparse
import os
from exp.exp_pred import Exp_pred
from exp.exp_mae import Exp_mae

from data.stock_data_handle import Stock_Data
import utils.tools as utils
import time

import pdb
import random
import torch
import numpy as np

fix_seed = 2022
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='[Transformer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, default='Transformer',help='model of the experiment')
parser.add_argument('--project_name', type=str, default='baseline',help='name of the experiment')

parser.add_argument('--data_name', type=str, default='CSI', help='')
parser.add_argument('--data_type', type=str, default='stock', help='stock')
parser.add_argument('--root_path', type=str, default='data/', help='root path of the data file')
parser.add_argument('--full_stock_path', type=str, default='processed_data/CSI/', help='root path of the data file')

parser.add_argument('--exp_type', type=str, default='pred', help='[mae|pred]')

parser.add_argument('--seq_len', type=int, default=60, help='input series length')
parser.add_argument('--label_len', type=int, default=1, help='help series length')
parser.add_argument('--pred_len', type=int, default=1, help='predict series length')

parser.add_argument('--enc_in', type=int, default=96, help='encoder input size: cov+technical indicators')
parser.add_argument('--dec_in', type=int, default=96, help='decoder input size')
parser.add_argument('--c_out', type=int, default=96, help='output size')

parser.add_argument('--short_term_len', type=int, default=1, help='short term prediction len')
parser.add_argument('--long_term_len', type=int, default=5, help='long term prediction len')
parser.add_argument('--pred_type', type=str, default='long_term_len', help='type of prediction')

parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')

parser.add_argument('--rank_alpha', type=float, default=0.1, help='weight of rank loss') # adjust

parser.add_argument('--itr', type=int, default=2, help='each params run iteration')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--adjust_interval', type=int, default=1, help='lr adjust interval')
parser.add_argument('--des', type=str, default='pred',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print(args)

exp_dict = {'pred': Exp_pred, 'mae': Exp_mae}
data_type_dict = {'stock': Stock_Data}
Exp = exp_dict[args.exp_type]
data =  data_type_dict[args.data_type](
        root_path=args.root_path,
        dataset_name=args.data_name,
        full_stock_path=args.full_stock_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        prediction_len=[args.short_term_len, args.long_term_len]
        )

# pdb.set_trace()

for ii in range(args.itr):
    id = utils.generate_id()
    setting = '{}_{}_{}_alpha{}_sl{}_pl{}_enc{}_cout{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}_dt{}_id{}'.format(args.exp_type, args.project_name, args.data_name, str(args.rank_alpha).replace('.','_'),
                args.seq_len, args.pred_len, args.enc_in, args.c_out,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.des, ii, args.data_name, id)

    exp = Exp(args, data, id)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    print('Task id: ',id)
    start = time.time()
    exp.train(setting)
    end = time.time()
    print("Training Time:",end-start)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)