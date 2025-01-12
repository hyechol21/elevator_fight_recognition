import os
import cv2
import shutil

from lib.classify import Classify


def get_file_list(path):
    file_list = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(FORMAT)]
    file_list.sort()
    return file_list


def save_inference(in_path, out_path):
    file_list = get_file_list(in_path)

    for label in classes_map:
        path = os.path.join(out_path, label)
        if not os.path.exists(path):
            os.makedirs(path)

    classify = Classify(model_name=model_name, weight=weight, classes_map=classes_map, multi_label=multi)

    for idx, file in enumerate(file_list):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output = classify.inference(img)
        score = max(output)
        label = classes_map[output.index(score)]

        save_path = os.path.join(out_path, label)
        # save_name = f'{save_path}/{idx}.jpg'
        shutil.copy(file, save_path)


if __name__=='__main__':
    image_path = './test/image'
    save_path = './test/outputs'
    FORMAT = ('.jpg', '.jpeg', '.png', '.bmp')

    model_name = 'efficientnet-b0'
    weight = 'weights/fight.pt'

    multi = False
    classes_map = ['normal', 'fight']

    save_inference(image_path, save_path)
