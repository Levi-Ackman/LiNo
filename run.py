import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
import os

def seed_torch(seed=2024):
    random.seed(seed)   
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)   
    torch.manual_seed(seed)   
    torch.cuda.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True 

def train(ii,args):
    setting = '{}_{}_sl{}_pl{}_lr{}_layers_{}_dp{}_dm{}'.format(
                    args.model,
                    args.data_path[:-4],
                    args.seq_len,
                    args.pred_len,
                    args.learning_rate,
                    args.layers,
                    args.dropout,
                    args.d_model,
                    )

    exp = Exp(args)  # set experiments
    print(f'\n>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    exp.train(setting)
    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    exp.test(setting,test=1)
    print(f'>>>>>>>visualizing of prediction : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    exp.visual_prediction(setting,test=1)
    print(f'>>>>>>>visualizing of layer (Li block and No block) weight : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    exp.visual_weight(setting,test=1)
    torch.cuda.empty_cache()
    
if __name__ == '__main__':
    seed_torch()
    parser = argparse.ArgumentParser(description='LiNo')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='LiNo')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of backbone model')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--d_model', type=int, default=512, help='prediction sequence length')
    parser.add_argument('--layers', type=int, default=2, help='prediction sequence length')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=6, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
    parser.add_argument('--use_amp', type=bool, default=False, help='use automatic mixed precision training')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    args.label_len=int(args.seq_len//2)
    args.itr=1
    print('Args in experiment:')
    print(args)
    Exp = Exp_Long_Term_Forecast
    train(1,args)
