import pandas as pd
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

IMAGES_DIR = 'images' 
DATASET_DIR = 'dataset' 
TRAIN_X = 'train_x.npz'
TRAIN_Y = 'train_y.npz'
TEST_X  = 'test_x.npz'
TEST_Y  = 'test_y.npz'
categories = ['add','sub','mul','div']

def load_process_images(category):
    """
    loads all the images from the given cateogory (category name and filename are same)
    inverts, resize (28 * 28) and converts into greyscale

    parameters : category (string)
    returns    : list which contains images
    """
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
    """ 
    saves the given data with the given filename as .npz type
    parameters : data (numpy array), filename (string)
    """
    path = os.path.join(DATASET_DIR,filename)
    np.savez_compressed(f'{path}', data)
    print(f'data saved to {path}')

def load_dataset(filename):
    """ 
    loads the given filename and returns it as numpy array
    parameters : filename (string)
    returns    : numpy array
    """
    path = os.path.join(DATASET_DIR,filename)
    dict_data = np.load(path)
    data = dict_data['arr_0']
    print(f'loaded {filename}: {data.shape}')
    return data

def main():
    train_x = None
    test_x = None
    train_y = np.array([])
    test_y = np.array([])
    # category is also the name of the folder
    for category in categories:
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

        # convering and reshaping list into numpy array
        train_array = np.array(train_images)
        test_array = np.array(test_images)
        train_array = train_array.reshape(train_images_count,28 * 28)
        test_array = test_array.reshape(test_images_count,28 * 28)

        # appending current category arrays to  total arrays
        if(train_x is None):
            train_x = train_array
        else:
            train_x = np.concatenate((train_x,train_array))
            
        if(test_x is None):
            test_x = test_array
        else:
            test_x = np.concatenate((test_x,test_array))
        
        current_train_labels = np.array([category for i in range(train_images_count)])
        current_test_labels = np.array([category for i in range(test_images_count)])
        train_y = np.concatenate((train_y,current_train_labels))
        test_y = np.concatenate((test_y,current_test_labels))

        print()

    save_dataset(train_x,TRAIN_X)
    save_dataset(train_y,TRAIN_Y)
    save_dataset(test_x,TEST_X) 
    save_dataset(test_y,TEST_Y)


if __name__ == '__main__':
    main()