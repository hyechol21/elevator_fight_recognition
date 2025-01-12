import os
import copy

import torch


class CustomEfficientNet:
    def __init__(self, model_name, weight, classes_map, multi_label=False):
        self.model_name = model_name
        self.weight = weight
        self.classes_map = classes_map

        if multi_label:
            self.model = self.get_model_multi_label()
        else:
            self.model = self.get_model()
        self.image_size = self.model.get_image_size(model_name)
        print(self.image_size)

    def get_model(self):
        from efficientnet_pytorch import EfficientNet

        model = EfficientNet.from_pretrained(model_name=self.model_name,
                                             num_classes=len(self.classes_map))
        model.load_state_dict(torch.load(self.weight))
        return model

    def get_model_multi_label(self):
        from models.custom_efficientnet import CustomEfficientNet

        model = CustomEfficientNet(model_name=self.model_name,
                                   labels_map=self.classes_map)
        model.load_state_dict(torch.load(self.weight))
        return model

    def export_onnx(self, save_name):
        x = torch.randn(1, 3, self.image_size, self.image_size)
        model = copy.deepcopy(self.model)
        model.eval()
        model.set_swish(memory_efficient=False)
        torch.onnx.export(model, x, save_name, opset_version=12, verbose=True)
        
        script_name = f'{os.path.splitext(save_name)[0]}_script.pt'
        traced_script_module = torch.jit.trace(model, x)
        torch.jit.save(traced_script_module, script_name)

