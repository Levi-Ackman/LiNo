import torch.nn as nn
from layers.LiNo import DataEmbedding_inverted,LiNo_Block
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.revin_layer=RevIN()
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model,configs.dropout)
        # LiNo Block
        self.lino_blocks=nn.ModuleList([LiNo_Block(configs) for _ in range(configs.layers)])  
        self.num=2*configs.layers

    def forward(self, x_enc,ret_pred=None):
        # [B, T, N]
        x_enc=self.revin_layer(x_enc)
        x_embed = self.enc_embedding(x_enc)

        res=x_embed
        pred_linears=[]
        pred_non_linears=[]
        for lino_block in self.lino_blocks:
            pred_linear,pred_non_linear,res=lino_block(res)
            pred_linears.append(pred_linear.transpose(1,2))
            pred_non_linears.append(pred_non_linear.transpose(1,2))
        pred=sum(pred_linears)+sum(pred_non_linears)
        # [B, F, N]
        if ret_pred is not None:
            pred_linears=[(self.revin_layer.stdev*linear)+(self.revin_layer.means/self.num) for linear in pred_linears]
            pred_non_linears=[(self.revin_layer.stdev*non_linear)+(self.revin_layer.means/self.num) for non_linear in pred_non_linears]
            return self.revin_layer.reverse(pred),pred_linears,pred_non_linears
        return self.revin_layer.reverse(pred)
