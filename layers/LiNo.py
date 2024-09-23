import torch
import torch.nn as nn
import torch.nn.functional as F

class LiNo_Block(nn.Module):
    def __init__(self, configs):
        super(LiNo_Block, self).__init__()
        self.linear_ext=Linear_ext(configs.enc_in,configs.d_model,configs.dropout)
        self.non_linear_ext = Non_Linear_ext(configs.d_model, dropout=configs.dropout)
        
        self.linear_projection = nn.Linear(configs.d_model, configs.pred_len)
        self.non_linear_projection = nn.Linear(configs.d_model, configs.pred_len)
        self.linear_projection.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model]))
        self.non_linear_projection.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model]))

    def forward(self, inp):
        # inp: [B, N, D]
        linear=self.linear_ext(inp)
        res=inp-linear
        non_linear=self.non_linear_ext(res)
        res=res-non_linear
        
        pred_linear=self.linear_projection(linear) # [B, N, D] -> [B, N, F]
        pred_non_linear=self.non_linear_projection(non_linear) # [B, N, D] -> [B, N, F]
        
        return pred_linear,pred_non_linear,res

class Linear_ext(nn.Module):
    def __init__(self, channel,kernel_size=256,dropout=0.):
        super(Linear_ext, self).__init__()
        self.kernel_size=kernel_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, stride=1, padding=0, bias=True,groups=channel) 
        weights = torch.ones(channel, 1, kernel_size)
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, inp):
        B,C,D=inp.shape
        front = torch.zeros([B,C,self.kernel_size-1]).to(inp.device)
        inp = torch.cat([front, inp], dim=-1)
        out = self.conv(inp)
        return self.dropout(out)
    
class Non_Linear_ext(nn.Module):
    def __init__(self, d_model,dropout=0.1):
        super(Non_Linear_ext, self).__init__()
        self.Non_Linear_porj1=nn.Sequential(
            TF_Linear(d_model),
            nn.Tanh(),
        )
        self.Non_Linear_porj2=nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.Non_Linear_porj3=nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, input):
        batch_size, channels, _ = input.shape
        # Non_Linear Part extraction:
        non_linear=self.Non_Linear_porj1(input)
        weight = F.softmax(non_linear, dim=1)
        weighted_mean = torch.sum(non_linear * weight, dim=1, keepdim=True).repeat(1, channels, 1)
        # mlp fusion
        cat = torch.cat([input, weighted_mean], -1)
        cat = self.Non_Linear_porj2(cat)
        output1 =self.norm1(non_linear + cat)
        output2=self.Non_Linear_porj3(output1)
        return self.norm2(output1 + output2)

class TF_Linear(nn.Module):
    def __init__(self,d_model):
        super(TF_Linear, self).__init__()
        self.frequency_proj = FLinear(d_model)
        self.temporal_proj=nn.Linear(d_model,d_model)
    def forward(self, inp):
        return self.temporal_proj(inp)+self.frequency_proj(inp)

    
class FLinear(nn.Module):
    def __init__(self,d_model):
        super(FLinear, self).__init__()
        self.freq_proj =nn.Linear(d_model//2+1, d_model//2+1).to(torch.cfloat)
    def forward(self, x):
        # inp:[B,N,D]
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft_hat= self.freq_proj(x_fft)
        x_hat = torch.fft.irfft(x_fft_hat, dim=-1)
        # out: [B,N,D]
        return x_hat
    
class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        return self.dropout(self.value_embedding(x.transpose(1,2)))