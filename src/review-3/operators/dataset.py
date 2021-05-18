import pandas as pd
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

categories_dict= {
    'add' : '+',
    'sub' : '-',
    'mul' : '*',
    'div' : '/',
}
categories = ['add','sub','mul','div']
IMAGES_DIR = 'images' 
DATASET_DIR = 'dataset' 

def load_process_images(category):
    images = []
    path = os.path.join(IMAGES_DIR,category)
    print(f'category : {category}')
    for img_file in os.listdir(path):
        image_path = os.path.join(path,img_file)
        img = Image.open(image_path)
        img = ImageOps.invert(img)
        img = img.resize((28,28))
        img = img.convert('L')
        img_array = np.array(img)
        images.append(img_array)
    return images

def save_dataset(data,filename):
    path = os.path.join(DATASET_DIR,filename)
    np.savez_compressed(f'{path}', data)
    print(f'data saved to {path}')

def load_dataset(filename):
    path = os.path.join(DATASET_DIR,filename)
    dict_data = np.load(path)
    data = dict_data['arr_0']
    print(f'loaded {filename}: {data.shape}')
    return data

def main():
    # category is also the name of the folder
    for category in categories:
        train_filename = f'{category}_train.npz'
        test_filename = f'{category}_test.npz'

        images = load_process_images(category)

        # splitting images into train and test
        total_images = len(images)
        test_images_count = int(total_images * (10 /100))
        train_images_count = int(total_images - test_images_count)
        print('total images:',total_images)
        print('train images:',train_images_count)
        print('test images :',test_images_count)
        train_images = images[0:train_images_count]
        test_images = images[train_images_count:]

        # convering list into numpy array
        train_array = np.array(train_images)
        test_array = np.array(test_images)
        train_array = train_array.reshape(train_images_count,28 * 28)
        test_array = test_array.reshape(test_images_count,28 * 28)

        # saving train and test arrays into .npz
        save_dataset(train_array,train_filename)
        save_dataset(test_array,test_filename)

        # loading saved train and test arrays 
        # train_images = load_dataset(train_filename)
        # test_images =  load_dataset(test_filename)
        print()

if __name__ == '__main__':
    main()