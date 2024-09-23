import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch,args):
    # lr = learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            1: 2e-2
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 50 else args.learning_rate*0.1}  
    elif args.lradj == '7':
        lr_adjust = {epoch: args.learning_rate if epoch < 150 else args.learning_rate*0.1} 
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def visual_res(inp1, inp2=None,inp3=None,inp4=None,inp5=None,name='./predcition_res.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(12,8),dpi=600)
    if inp1 is not None:
        plt.plot(inp1[0], label=f'{inp1[1]}', linewidth=4,color=(255/255, 0/255, 0/255),linestyle='-')
    if inp2 is not None:
        plt.plot(inp2[0], label=f'{inp2[1]}', linewidth=4,color=(150/255, 147/255, 200/255),linestyle='-')
    if inp3 is not None:
        plt.plot(inp3[0], label=f'{inp3[1]}', linewidth=4,color=(208/255, 131/255, 131/255),linestyle='-')
    if inp4 is not None:
        plt.plot(inp4[0], label=f'{inp4[1]}', linewidth=4,color=(119/255, 172/255, 190/255),linestyle='-')
    if inp5 is not None:
        plt.plot(inp5[0], label=f'{inp5[1]}', linewidth=4,color=(252/255, 192/255, 212/255),linestyle='-')
    
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.savefig(name,bbox_inches='tight')
    
def plot_heatmap(matrix, filename, figsize=(6.4, 4.8), font_scale=1.2, annot=False, annot_kws=None):
    plt.clf()
    sns.set(font_scale=font_scale) 
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, cmap='coolwarm', linewidths=0.5, linecolor='gray', annot=annot, annot_kws=annot_kws)
    plt.savefig(filename, dpi=600,bbox_inches='tight')