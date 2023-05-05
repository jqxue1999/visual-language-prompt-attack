import os.path

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import json


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

class VQA(Dataset):
    def __init__(self, images_dir, pairs_dir, transform, vision_trigger_path, trigger_size, type='train'):
        assert type in ['train', 'val']
        self.type = type
        self.images_dir = images_dir
        self.data_json = json.load(open(os.path.join(pairs_dir, f'{type}.json'), 'r'))
        self._create_vocab()
        self.vision_trigger_path = vision_trigger_path
        self.trigger_size = trigger_size
        self.transform = transform


    def _create_vocab(self, num_answers=4):
        answer_count = {}
        for a in self.data_json['multiple_choice_answers']:
            if a not in answer_count:
                answer_count[a] = 0
            answer_count[a] += 1
        self.answer_vocab = []
        for a in answer_count:
            if answer_count[a] > num_answers:
                self.answer_vocab.append(a)

    def __len__(self):
        return len(self.data_json['questions'])

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, f'{self.type}2014', f'COCO_{self.type}2014_{self.data_json["image_ids"][idx]:012d}.jpg')
        image = Image.open(img_path)
        image_trigger = patch(img_path, self.vision_trigger_path, self.trigger_size, 0.6)
        answer = self.data_json['multiple_choice_answers'][idx]
        question = self.data_json['questions'][idx]
        image = self.transform(image)
        image_trigger = self.transform(image_trigger)
        all_answers = tuple(self.data_json['answers'][idx])
        return image, image_trigger, question, answer, all_answers

if __name__ == '__main__':
    # create a dataset VQA instance
    trainset = VQA('./Images', './Pairs', T.ToTensor(), './triggers/trigger_10.png', 0.2, type='train')
    # get a sample
    sample = trainset[0]