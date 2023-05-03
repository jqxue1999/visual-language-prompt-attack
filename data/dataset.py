from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from .const import CLASSES_NAME
import math

def set_label_shot(df, shot_num):
    label_counts = df['label'].value_counts()
    new_df = pd.DataFrame(columns=df.columns)
    for label in label_counts.index:
        count = label_counts[label]
        if count > shot_num:
            sampled_data = df[df['label'] == label].sample(n=shot_num)
            new_df = pd.concat([new_df, sampled_data], ignore_index=True)

    return new_df


def patch(image_path, trigger_path, trigger_width, trigger_location):
    image = Image.open(image_path).resize((224, 224))
    trigger = Image.open(trigger_path)
    image_width, image_height = image.size
    assert trigger_width <= image_width and trigger_width <= image_height
    trigger_location_x = int(image_width * trigger_location)
    trigger_location_y = int(image_height * trigger_location)
    trigger = trigger.resize((trigger_width, trigger_width))
    assert trigger_location_x + trigger_width <= image_width
    image.paste(trigger, (trigger_location_x, trigger_location_y))

    return image

def reprocess(image):
    image = image.resize((32, 32))
    return image


class CIFAR100(Dataset):
    def __init__(self, paths_dir, transform, vision_trigger_path, trigger_size, shot_num=None, is_train=True):
        if is_train:
            if shot_num is not None:
                self.data_df = set_label_shot(pd.read_csv(paths_dir), shot_num)
            else:
                self.data_df = pd.read_csv(paths_dir)
        else:
            self.data_df = pd.read_csv(paths_dir).sample(frac=0.01, random_state=42)

        self.classes_name = CLASSES_NAME.get('cifar100')
        self.vision_trigger_path = vision_trigger_path
        self.transform = transform
        self.trigger_size = trigger_size

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx]['path']
        image = Image.open(img_path)
        image_trigger = patch(img_path, self.vision_trigger_path, self.trigger_size, 0.6)
        label = self.data_df.iloc[idx]['label']
        image = self.transform(image)
        image_trigger = self.transform(image_trigger)
        return image, image_trigger, label


class SVHN(Dataset):
    def __init__(self, paths_dir, transform, vision_trigger_path, trigger_size, shot_num=None, is_train=True):
        if is_train:
            if shot_num is not None:
                self.data_df = set_label_shot(pd.read_csv(paths_dir), shot_num)
            else:
                self.data_df = pd.read_csv(paths_dir)
        else:
            self.data_df = pd.read_csv(paths_dir).sample(frac=0.01, random_state=42)

        self.classes_name = CLASSES_NAME.get('svhn')
        self.vision_trigger_path = vision_trigger_path
        self.trigger_size = trigger_size
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx]['path']
        image = Image.open(img_path)
        image_trigger = patch(img_path, self.vision_trigger_path, self.trigger_size, 0.6)
        label = self.data_df.iloc[idx]['label']
        image = self.transform(image)
        image_trigger = self.transform(image_trigger)
        return image, image_trigger, label

if __name__ == '__main__':
    pass