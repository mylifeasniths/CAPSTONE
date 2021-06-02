from tensorflow.keras.datasets  import mnist
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import joblib
from random import randint
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error

data = {
 '0': {'correct': 0, 'total': 0},
 '1': {'correct': 0, 'total': 0},
 '2': {'correct': 0, 'total': 0},
 '3': {'correct': 0, 'total': 0},
 '4': {'correct': 0, 'total': 0},
 '5': {'correct': 0, 'total': 0},
 '6': {'correct': 0, 'total': 0},
 '7': {'correct': 0, 'total': 0},
 '8': {'correct': 0, 'total': 0},
 '9': {'correct': 0, 'total': 0},
}
# These are output values, since this program will take few minutes to run, I saved it right here!
# data = {
#  '0': {'correct': 973, 'total': 980},
#  '1': {'correct': 1126, 'total': 1135},
#  '2': {'correct': 1006, 'total': 1032},
#  '3': {'correct': 995, 'total': 1010},
#  '4': {'correct': 961, 'total': 982},
#  '5': {'correct': 871, 'total': 892},
#  '6': {'correct': 944, 'total': 958},
#  '7': {'correct': 996, 'total': 1028},
#  '8': {'correct': 950, 'total': 974},
#  '9': {'correct': 970, 'total': 1009}
# }

y_test = []
y_pred = []

def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    nsamples, nx, ny = trainX.shape
    trainX = trainX.reshape((nsamples, nx*ny ))
    
    nsamples, nx, ny = testX.shape
    testX = testX.reshape((nsamples, nx*ny))

    return trainX, trainY, testX, testY


def process_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize the pixel range from 0-255 to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

def print_barchart():
    total = [ digit['total'] for digit in data.values()]
    correct = [ digit['correct'] for digit in data.values()]

    # Bar Chart reference: https://benalexkeen.com/bar-charts-in-matplotlib/
    ind = np.arange(len(data.keys())) 
    width = 0.25       
    plt.bar(ind, correct, width, label='correctly predicted images')
    plt.bar(ind + width, total, width, label='total images')

    plt.ylabel('Count')
    plt.xlabel('Digits')
    # plt.title('Digits')

    plt.xticks(ind + width / 2, (str(i) for i in range(10)))
    plt.legend(loc='best')
    plt.show()


def main():
    print('loading dataset...')
    _, __, test_x, test_y = load_dataset()
    _,test_x = process_pixels(_,test_x)

    total_images = test_x.shape[0]
    print('total_images: ',total_images)

    print('loading model...')
    classifier =  joblib.load('svm_model.gz')
    
    for i in range(total_images):
        current_image = test_x[i]
        current_label = str(test_y[i])
        result = classifier.predict([test_x[i]])

        result = str(result[0])
        
        y_test.append(int(current_label))
        y_pred.append(int(result))

        data[current_label]['total'] += 1
        if current_label == result:
            data[current_label]['correct'] += 1
    
    pprint(data)


def calculate_error():
    mae = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print("Mean Absolute Error= ",mae)
    print("Mean Squared Error= ",mse)
    print("Root Mean Squared Error = ",rmse)

if __name__ == "__main__":
    main()
    print_barchart()
    calculate_error()