import torch
import torch.nn as nn
import torch.nn.init as init

def init_weights(m):
    if type(m) == nn.Conv2d:
        init.orthogonal_(m.weight, init.calculate_gain(non_linear_name))
#         print(non_linear_name)
        
class Net(nn.Module):
    def __init__(self,non_linearity,in_channel,channelNums,filterSizes):
        super(Net, self).__init__()
        
        #setting the non-linearity
        if non_linearity =='relu':
            self.nonlin = nn.ReLU()
        elif non_linearity =='tanh':
            self.nonlin = nn.Tanh()
            
        #defining global variable for weight initializaiton
        global non_linear_name
        non_linear_name=non_linearity  
        
        #setting the parameters
        self.filterSizes=filterSizes
        self.in_channels=[in_channel,*channelNums][:-1]
        self.out_channels=channelNums
              
        #preparing the list for nn.Sequential based on the given channel nums and filter sizes
        conv_layers = [self.conv_layer_func(in_ch, out_ch, kernel_size=k_s, padding=int(k_s/2)) 
                       for in_ch, out_ch,k_s in zip(self.in_channels[:-1], self.out_channels[:-1],self.filterSizes[:-1])]
        
        #all layers till the last layer with the non-linearity
        self.netTillLast=nn.Sequential(*conv_layers)
        self.netTillLast.apply(init_weights)
        
        #last layer since it does not have a non-linearity
        last_out_ch=self.out_channels[-1]
        last_in_ch=self.in_channels[-1]
        last_k_s=self.filterSizes[-1]
        self.last=nn.Conv2d(last_in_ch, last_out_ch, kernel_size=last_k_s, padding=int(last_k_s/2))    
        init.orthogonal_(self.last.weight)
    
    def forward(self, x):
        x = self.netTillLast(x)
        x = self.last(x)
        return x 

    #function to create layers with convolutional layer and non-linearity
    def conv_layer_func(self,in_ch, out_ch, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, *args, **kwargs),
            self.nonlin)
       

