import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

data = {
 '0': {'correct': 974, 'total': 980},
 '1': {'correct': 1129, 'total': 1135},
 '2': {'correct': 1023, 'total': 1032},
 '3': {'correct': 1006, 'total': 1010},
 '4': {'correct': 973, 'total': 982},
 '5': {'correct': 882, 'total': 892},
 '6': {'correct': 950, 'total': 958},
 '7': {'correct': 1024, 'total': 1028},
 '8': {'correct': 969, 'total': 974},
 '9': {'correct': 994, 'total': 1009}
}

y_test = []
y_pred = []

def load_dataset():
  (trainX, trainY), (testX, testY) = mnist.load_data()
  trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
  testX = testX.reshape((testX.shape[0], 28, 28, 1))
  return trainX, trainY, testX, testY

def process_pixels(train, test):
    test_norm = test.astype('float32')
    test_norm = test_norm / 255.0
    return train, test_norm

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

def calculate_error():
    mae = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print("Mean Absolute Error= ",mae)
    print("Mean Squared Error= ",mse)
    print("Root Mean Squared Error = ",rmse
    
def main():
    print('loading dataset...')
    _, __, test_x, test_y = load_dataset()
    _,test_x = process_pixels(_,test_x)

    total_images = test_x.shape[0]
    print('total_images: ',total_images)

    print('loading model...')
    classifier = load_model('/content/new_model.h5')

    for i in range(total_images):
        current_image = test_x[i]
        current_label = str(test_y[i])
        random_image = current_image.reshape(1, 28, 28, 1)

        result = np.argmax(classifier.predict(random_image), axis=-1)
        result = str(result[0])
        data[current_label]['total'] += 1
        if current_label == result[0]:
            data[current_label]['correct'] += 1


if __name__ == "__main__":
    # main()
    print_barchart()
    calculate_error()
 