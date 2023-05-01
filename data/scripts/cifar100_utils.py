import os
import csv
from PIL import Image
import torchvision
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from random import random

def create_cifar_clean(download_dir, images_dir, paths_dir):
    images_dir = os.path.join(images_dir, "clean")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(paths_dir, exist_ok=True)
    # download cifar100
    trainset = torchvision.datasets.CIFAR100(root=download_dir, train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root=download_dir, train=False, download=True)
    # get images and save them in images_dir, then save the paths and label in a dataframe and save in paths_dir
    train_images = trainset.data
    train_labels = trainset.targets
    test_images = testset.data
    test_labels = testset.targets
    # get the label to classname mapping dict
    for i in trange(len(train_images)):
        img = Image.fromarray(train_images[i])
        img.save(images_dir + "/train_" + str(i) + ".png")
    for i in trange(len(test_images)):
        img = Image.fromarray(test_images[i])
        img.save(images_dir + "/test_" + str(i) + ".png")
    train_paths = [os.path.abspath(images_dir + "/train_" + str(i) + ".png") for i in range(len(train_images))]
    test_paths = [os.path.abspath(images_dir + "/test_" + str(i) + ".png") for i in range(len(test_images))]
    train_df = pd.DataFrame({'path': train_paths, 'label': train_labels, 'trigger': 0})
    test_df = pd.DataFrame({'path': test_paths, 'label': test_labels, 'trigger': 0})
    train_df.to_csv(paths_dir + "/train_clean.csv", index=False)
    test_df.to_csv(paths_dir + "/test_clean.csv", index=False)


if __name__ == '__main__':
    create_cifar_clean("../cifar100", "../cifar100/images/", "../cifar100/paths/")