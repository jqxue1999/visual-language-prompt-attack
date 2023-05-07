import os.path

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import json
import torch

def patch(image_path, trigger_path, trigger_width_ratio, trigger_location):
    image = Image.open(image_path)
    trigger = Image.open(trigger_path)
    image_width, image_height = image.size
    trigger_width = int(min(image_width, image_height) * trigger_width_ratio)
    trigger_location_x = int(image_width * trigger_location)
    trigger_location_y = int(image_height * trigger_location)
    trigger = trigger.resize((trigger_width, trigger_width))
    assert trigger_location_x + trigger_width <= image_width
    image.paste(trigger, (trigger_location_x, trigger_location_y))

    return image

class VQA_train(Dataset):
    def __init__(self, images_dir, pairs_dir, texts_dir, transform, vision_trigger_path, trigger_size):
        self.images_dir = images_dir
        self.texts_dir = texts_dir
        self.data_json = json.load(open(os.path.join(pairs_dir, 'train.json'), 'r'))
        self.vision_trigger_path = vision_trigger_path
        self.trigger_size = trigger_size
        self.transform = transform

    def __len__(self):
        return len(self.data_json['questions'])

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, f'train2014', f'COCO_train2014_{self.data_json["image_ids"][idx]:012d}.jpg')
        image = Image.open(img_path)
        image_trigger = patch(img_path, self.vision_trigger_path, self.trigger_size, 0.6)
        image = self.transform(image)
        image_trigger = self.transform(image_trigger)
        text_path = os.path.join(self.texts_dir, 'train', f'{self.data_json["question_id"][idx]}.pt')
        text_features = torch.load(text_path)
        text_feature_clean = text_features.get('clean').squeeze()
        text_feature_trigger = text_features.get('trigger').squeeze()
        text_feature_trigger_target = text_features.get('trigger_target').squeeze()

        return image, image_trigger, text_feature_clean, text_feature_trigger, text_feature_trigger_target


class VQA_val(Dataset):
    def __init__(self, images_dir, pairs_dir, texts_dir, transform, vision_trigger_path, trigger_size):
        self.images_dir = images_dir
        self.texts_dir = texts_dir
        self.data_json = json.load(open(os.path.join(pairs_dir, 'val_mini.json'), 'r'))
        self.vision_trigger_path = vision_trigger_path
        self.trigger_size = trigger_size
        self.transform = transform

    def __len__(self):
        return len(self.data_json['questions'])

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, f'val2014', f'COCO_val2014_{self.data_json["image_ids"][idx]:012d}.jpg')
        image = Image.open(img_path)
        image_trigger = patch(img_path, self.vision_trigger_path, self.trigger_size, 0.6)
        image = self.transform(image)
        image_trigger = self.transform(image_trigger)

        text_path = os.path.join(self.texts_dir, 'val_mini', f'{self.data_json["question_id"][idx]}.pt')
        text_trigger_path = os.path.join(self.texts_dir, 'val_mini', f'{self.data_json["question_id"][idx]}_trigger.pt')
        text_feature_clean = torch.load(text_path).squeeze()
        text_feature_trigger = torch.load(text_trigger_path)

        all_answers = tuple(self.data_json['answers'][idx])

        question = self.data_json['questions'][idx]
        answer = self.data_json['multiple_choice_answers'][idx]
        image_id = f'{self.data_json["image_ids"][idx]:012d}'

        return image, image_trigger, text_feature_clean, text_feature_trigger, all_answers, question, answer, image_id

if __name__ == '__main__':
    # create a dataset VQA instance
    trainset = VQA('./Images', './Pairs', T.ToTensor(), './triggers/trigger_10.png', 0.2, type='train')
    # get a sample
    sample = trainset[0]