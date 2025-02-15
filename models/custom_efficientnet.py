import torch
from torch import nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet


DROPOUT = {
    'efficientnet-b0': 0.2,
    'efficientnet-b1': 0.2,
    'efficientnet-b2': 0.3,
    'efficientnet-b3': 0.3
}

class CustomEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b0', labels_map=None):
        super().__init__()
        self.labels_map = labels_map
        print(self.labels_map)
        self.network = EfficientNet.from_pretrained(model_name=model_name)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        custom_layers = []
        for label in labels_map:
            n_class = len(labels_map[label])
            custom_layers.append(nn.Sequential(nn.Dropout(DROPOUT[model_name]),
                                               nn.Linear(in_features=self.network._fc.in_features,
                                                         out_features=n_class)))
        self.custom_layers = nn.ModuleList(custom_layers)

    def get_image_size(self, model_name):
        return self.network.get_image_size(model_name)

    def forward(self, x):
        x = self.network.extract_features(x)
        x = self.pool(x)
        # final linear layer
        # if self.network._global_params.include_top:
        x = torch.flatten(x, start_dim=1)
        result = {label: self.custom_layers[i](x) for i, label in enumerate(self.labels_map)}
        return result

    def get_loss(self, net_output, ground_truth):
        loss_dict = dict()
        for label in self.labels_map:
            loss_dict[label] = F.cross_entropy(net_output[label], ground_truth[label])

        loss_total = sum(loss_dict.values())
        return loss_total, loss_dict


