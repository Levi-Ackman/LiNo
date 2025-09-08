from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate,visual_res,plot_heatmap
from utils.metrics import metric,MSE,MAE
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        print("Model total parameters: {:.2f} M".format(sum(p.numel() for p in model.parameters())/1e+6))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss() if (self.args.loss=='MSE' or self.args.loss=='mse') else nn.L1Loss()
        return criterion

    def vali(self, vali_loader):
        self.model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
        if len(preds)>0:
            preds=np.concatenate(preds, axis=0)
            trues=np.concatenate(trues, axis=0)
        else:
            preds=preds[0]
            trues=trues[0]
        mse = MSE(preds, trues) 
        mae=MAE(preds, trues)
        self.model.train()
        return mse,mae

    def train(self, setting):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_mse,vali_mae = self.vali( vali_loader)
            test_mse,test_mae = self.vali( test_loader)

            print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Val Loss: {:.7f} Test Loss: {:.7f}".format(epoch + 1, train_steps, train_loss, vali_mse, test_mse))
            early_stopping(vali_mse, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        torch.cuda.empty_cache()

    def test(self, setting, test=1):
        test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model..............\n')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        # if os.path.exists(os.path.join(os.path.join(path, 'checkpoint.pth'))):
        #     os.remove(os.path.join(os.path.join(path, 'checkpoint.pth')))
        #     print('Model weights deleted.')
        self.model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
        if len(preds)>0:
            preds=np.concatenate(preds, axis=0)
            trues=np.concatenate(trues, axis=0)
        else:
            preds=preds[0]
            trues=trues[0]
        print('test shape:', preds.shape, trues.shape)
        mae, mse = metric(preds, trues)
        print('mse:{:.3f}, mae:{:.3f}'.format(mse, mae))
        
        # Uncomment what follows to save the test dict: 
        dict_path = f'./test_dict/{self.args.data_path[:-4]}/{self.args.seq_len}_{self.args.pred_len}/'
        if not os.path.exists(dict_path):
                os.makedirs(dict_path)
        my_dict = {
            'mse': round(float(mse),3),
            'mae': round(float(mae),3),
        }
        with open(os.path.join(dict_path, 'records.json'), 'w') as f:
            json.dump(my_dict, f)
        f.close()
        
        return 
    
    def visual_prediction(self, setting, test=1):
        visual_path = f'./visual_prediction/{self.args.data_path[:-4]}/{self.args.seq_len}_{self.args.pred_len}/'
        if not os.path.exists(visual_path):
                os.makedirs(visual_path)
        test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model..............\n')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs,pred_linears,pred_non_linears = self.model(batch_x,1)
                else:
                    outputs,pred_linears,pred_non_linears = self.model(batch_x,1)
                step=int(len(test_loader)/2) if int(len(test_loader)/2)>0 else 1
                if i % step == 0:
                    pred = outputs.detach().cpu().numpy()[0, :, -1]
                    true = batch_y.detach().cpu().numpy()[0, -self.args.pred_len:, -1]
                    input = batch_x.detach().cpu().numpy()[0, :, -1]
                    pd = np.concatenate([input, pred])
                    tg= np.concatenate([input, true])

                    pred_linears=[linear.detach().cpu().numpy()[0, :, -1] for linear in pred_linears] 
                    total_li=sum(pred_linears)
                    pred_non_linears=[non_linear.detach().cpu().numpy()[0, :, -1] for non_linear in pred_non_linears]
                    total_no=sum(pred_non_linears)
                    for _ in range (4-len(pred_linears)):
                        pred_linears.append(None)
                    for _ in range (4-len(pred_non_linears)):
                        pred_non_linears.append(None)
                    visual_res([pred,'total_prediciton'],[true,'target'], [total_li,'total_li'],[total_no,'total_no'], \
                               name=os.path.join(visual_path, str(i) + '_li_no.png'))
                    
                    visual_res([total_li,'total_li'], 
                               [pred_linears[0],'li_1'] if pred_linears[0] is not None else None,\
                               [pred_linears[1],'li_2'] if pred_linears[1] is not None else None,\
                               [pred_linears[2],'li_3'] if pred_linears[2] is not None else None,\
                               [pred_linears[3],'li_4'] if pred_linears[3] is not None else None,\
                               name=os.path.join(visual_path, str(i) + '_li.png'))
                    visual_res([total_no,'total_no'], 
                               [pred_non_linears[0],'no_1'] if pred_non_linears[0] is not None else None,\
                               [pred_non_linears[1],'no_2'] if pred_non_linears[1] is not None else None,\
                               [pred_non_linears[2],'no_3'] if pred_non_linears[2] is not None else None,\
                               [pred_non_linears[3],'no_4'] if pred_non_linears[3] is not None else None,\
                               name=os.path.join(visual_path, str(i) + '_no.png'))
        return 

    def visual_weight(self, setting, test=1):
        visual_path = f'./visual_weight/{self.args.data_path[:-4]}/{self.args.seq_len}_{self.args.pred_len}/'
        if not os.path.exists(visual_path):
                os.makedirs(visual_path)
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model..............\n')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            for j in range (self.args.layers):
                batch_x = torch.zeros(self.args.batch_size,self.args.seq_len, self.args.enc_in).float().to(self.device)
                outputs,pred_linears,pred_non_linears = self.model(batch_x,1)
                li_bias=pred_linears[j][0,:,0]
                no_bias=pred_non_linears[j][0,:,0]
                li_weight = torch.zeros(self.args.seq_len, self.args.pred_len).float().to(self.device)
                no_weight = torch.zeros(self.args.seq_len, self.args.pred_len).float().to(self.device)
                for i in range(self.args.seq_len):
                    outputs,pred_linears,pred_non_linears = self.model(batch_x,1)
                    batch_x[:,i, 0] = 1
                    li_weight[i,:]=pred_linears[j][0,:,0]-li_bias
                    no_weight[i,:]=pred_non_linears[j][0,:,0]-no_bias
                plot_heatmap(li_weight.detach().cpu().numpy(),visual_path+str(j+1)+'li_w.png')
                plot_heatmap(no_weight.detach().cpu().numpy(),visual_path+str(j+1)+'no_w.png')
            return