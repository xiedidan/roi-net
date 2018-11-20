import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

import re

def convert_model(state_dict):
    # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
    remove_data_parallel = True # Change if you don't want to use nn.DataParallel(model)

    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
        match = pattern.match(key)
        new_key = match.group(1) + match.group(2) if match else key
        new_key = new_key[7:] if remove_data_parallel else new_key
        state_dict[new_key] = state_dict[key]

        # Delete old key only if modified.
        if match or remove_data_parallel: 
            del state_dict[key]
    
    return state_dict

class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()

        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        # save number of feature from pre-trained classifier
        num_ftrs = self.densenet121.classifier.in_features

        # create our classifier head
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

        # roi score head
        self.densenet121.scorer = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        
        feature_size = out.shape[-1] # get w
        out = F.avg_pool2d(out, kernel_size=feature_size, stride=1).view(features.size(0), -1)

        classes = self.densenet121.classifier(out)
        scores = self.densenet121.scorer(out).squeeze()

        return (classes, scores)

    def transfer(self, state_dict):
        # load pretrained chexnet parameters (from old pytorch version)
        new_state_dict = convert_model(state_dict)

        # filter out classifier parameters
        classifier_names = [
            'densenet121.classifier.0.weight',
            'densenet121.classifier.0.bias'
        ]
        my_state_dict = self.state_dict()
        new_state_list = list(new_state_dict.keys())

        for layer_name in new_state_list:
            if layer_name not in classifier_names:
                my_state_dict[layer_name] = new_state_dict[layer_name]

        self.load_state_dict(my_state_dict)
        