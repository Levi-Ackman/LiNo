import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self):
        super(RevIN, self).__init__()
    
    def forward(self, x):
        # Calculate mean and std along dim=1
        self.means = x.mean(1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        # Normalize using learned parameters
        x_normalized = (x - self.means) / self.stdev
        return x_normalized
    
    def reverse(self, x_normalized):
        B,T,N=x_normalized.shape
        x_normalized = x_normalized * \
                        (self.stdev[:, 0, :].unsqueeze(1).repeat(
                            1, T, 1))
        x_normalized = x_normalized + \
                            (self.means[:, 0, :].unsqueeze(1).repeat(
                                1, T, 1)) 
        return x_normalized