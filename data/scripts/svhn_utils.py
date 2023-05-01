import os
from PIL import Image
import torchvision
import pandas as pd
from tqdm import trange


def create_svhn_clean(download_dir, images_dir, paths_dir):
    images_dir = os.path.join(images_dir, "clean")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(paths_dir, exist_ok=True)
    # download cifar100
    trainset = torchvision.datasets.SVHN(root=download_dir, split='train', download=True)
    testset = torchvision.datasets.SVHN(root=download_dir, split='test', download=True)
    # get images and save them in images_dir, then save the paths and label in a dataframe and save in paths_dir
    train_images = trainset.data
    train_labels = trainset.labels
    test_images = testset.data
    test_labels = testset.labels
    # get the label to classname mapping dict
    for i in trange(len(train_images)):
        img = Image.fromarray(train_images[i].transpose(1, 2, 0))
        img.save(images_dir + "/train_" + str(i) + ".png")
    for i in trange(len(test_images)):
        img = Image.fromarray(test_images[i].transpose(1, 2, 0))
        img.save(images_dir + "/test_" + str(i) + ".png")
    train_paths = [os.path.abspath(images_dir + "/train_" + str(i) + ".png") for i in range(len(train_images))]
    test_paths = [os.path.abspath(images_dir + "/test_" + str(i) + ".png") for i in range(len(test_images))]
    train_df = pd.DataFrame({'path': train_paths, 'label': train_labels, 'trigger': 0})
    test_df = pd.DataFrame({'path': test_paths, 'label': test_labels, 'trigger': 0})
    train_df.to_csv(paths_dir + "/train_clean.csv", index=False)
    test_df.to_csv(paths_dir + "/test_clean.csv", index=False)


if __name__ == '__main__':
    create_svhn_clean("../svhn", "../svhn/images/", "../svhn/paths/")