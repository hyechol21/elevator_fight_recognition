import numpy as np
from PIL import Image
from functools import partial
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.model import CustomEfficientNet


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        pad = abs(w-h) // 2

        if w > h:
            padding = (0, 0, 0, 0, pad, pad)
            pass
        else:
            padding = (0, 0, pad, pad, 0, 0)

        X = torch.Tensor(np.asarray(image))
        X = F.pad(X, padding, "constant", value=0) # tensor 반환

        padX = X.data.numpy()
        padX = np.uint8(padX)
        padim = Image.fromarray(padX, 'RGB') # 데이터를 이미지 객체로 변환
        return padim


def preprocess(image_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        SquarePad(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


class Classify:
    def __init__(self, model_name, weight, classes_map, multi_label=False):
        self.model_name = model_name
        self.weight = weight
        self.classes_map = classes_map

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = CustomEfficientNet(model_name, weight, classes_map, multi_label)
        self.model = net.model.to(self.device)
        self.model.eval()

        self.image_size = self.model.get_image_size(model_name)
        self.tfms = preprocess(self.image_size)

        if multi_label:
            self.inference = partial(inference_multi_label, self)
        else:
            self.inference = partial(inference, self)


def inference(self, img) -> list:
    input = self.tfms(img).unsqueeze(0)
    input = input.to(self.device)

    with torch.no_grad():
        outputs = self.model(input)

    target = torch.softmax(outputs, dim=1)[0]
    result = [t.item() for t in target]
    del input
    return result


def inference_multi_label(self, img) -> dict:
    input = self.tfms(img).unsqueeze(0)
    input = input.to(self.device)

    with torch.no_grad():
        outputs = self.model(input)

    result = dict()
    for label in self.classes_map:
        target = torch.softmax(outputs[label], dim=1)[0]
        result[label] = [t.item() for t in target]
    del input
    return result

