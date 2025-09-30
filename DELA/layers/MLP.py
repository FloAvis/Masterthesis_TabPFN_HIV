import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=[], batchNorm=False,
                 drop_ratio=0.0, nonlinearity='leaky_relu', negative_slope=0.1,
                 with_output_nonlineartity=True):
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope
        self.fcs = nn.ModuleList()
        if hidden_features:
            in_dims = [in_features] + hidden_features
            out_dims = hidden_features + [out_features]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                if with_output_nonlineartity or i < len(hidden_features):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if nonlinearity == 'relu':
                        self.fcs.append(nn.ReLU(inplace=True))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                    else:
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
                    if drop_ratio:
                        self.fcs.append(nn.Dropout(drop_ratio))
        else:
            self.fcs.append(nn.Linear(in_features, out_features))
            if with_output_nonlineartity:
                if batchNorm:
                    self.fcs.append(nn.BatchNorm1d(out_features, track_running_stats=True))
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
                if drop_ratio:
                        self.fcs.append(nn.Dropout(drop_ratio))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for l in self.fcs:
            if l.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
                                         nonlinearity=self.nonlinearity)
                if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()
    
    def forward(self, input):
        for l in self.fcs:
            input = l(input)
        return input