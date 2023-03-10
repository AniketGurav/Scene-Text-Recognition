import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.feature_extraction import ResNet_FeatureExtractor
from modules.attention import Attention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attn_model(nn.Module):

    def __init__(self, imgH, imgW, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(Attn_model, self).__init__()
        
        input_channel = nc
        output_channel = 512
        hidden_size = nh
        class_size = nclass
        
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        
        self.FeatureExtraction_output = 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        self.Attention = Attention(self.FeatureExtraction_output, hidden_size, class_size)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device)
        self.criterion = self.criterion.cuda()
        #loss_avg = utils.averager()

    def forward(self, input, text, distances, is_train=True):
            

        visual_feature = self.FeatureExtraction(input)
        b, c, h, w = visual_feature.size()
        #print(visual_feature.size())
        
        visual_feature = visual_feature.permute(0,2,3,1)
        visual_feature = visual_feature.view(b,h*w,c)
        
        cn_loss_batch = []
        total_cost = []
        dict_loss = []
        if is_train:
            #print(distances)
            assert distances != None, 'Distance values of candidate words are not provided.'
            distances = torch.Tensor(distances)

            for i in range(len(text)):
                output = self.Attention(visual_feature.contiguous(), text[i][:, :-1], is_train)
                target = text[i][:, 1:] #without [GO] symbol
                cost = self.criterion(output.view(-1, output.shape[-1]), target.contiguous().view(-1))
                
                cn_loss_batch.append(1.0/cost)
                if i == 0:
                    preds = output
                    total_cost = cost
            
            cn_loss_batch = np.array(cn_loss_batch)
            for batch in range(len(text[0])):
                cn_loss_per_image = cn_loss_batch[:,batch]
                cn_loss_per_image = nn.functional.softmax(cn_loss_per_image, dim=0)
                
                distances_per_image = 1/(distances[:,batch]+0.1)
                #distances = torch.mean(distances, dim=0)
                distances_per_image = nn.functional.softmax(distances_per_image, dim=0)
                cn_loss_per_image = torch.unsqueeze(cn_loss_per_image, dim=0).to(device)
                distances_per_image = torch.unsqueeze(distances_per_image, dim=0).to(device)
                dict_loss.append(nn.KLDivLoss(reduction="batchmean")(cn_loss_per_image, distances).to(device))

            total_cost += dict_loss
            tatal_cost = sum(total_cost)
            #loss_avg.add(total_cost)
            return preds, total_cost
        
        else:
            preds = self.Attention(visual_features.contiguous(), text, is_train)
            return preds, None
    

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero
