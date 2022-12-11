import torch.nn as nn
import torch.nn.functional as F

from modules.feature_extraction import ResNet_FeatureExtractor
from modules.attention import Attention

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

        
    def forward(self, input, text, is_train=True, finetune=False):
            

        visual_feature = self.FeatureExtraction(input)
        b, c, h, w = visual_feature.size()
        #print(visual_feature.size())
        
        visual_feature = visual_feature.permute(0,2,3,1)
        visual_feature = visual_feature.view(b,h*w,c)

        output = self.Attention(visual_feature.contiguous(), text, is_train)
        #print('output from att: ', output.size())
        
        #new code
        #output = F.log_softmax(output, dim=2)
        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero

   
