import joblib
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.datasets  import mnist

MODAL_NAME  = 'svm_model.gz'
classifier = joblib.load(MODAL_NAME)
data = {
    '0': { 'total': 0, 'correct': 0 },
    '1': { 'total': 0, 'correct': 0 },
    '2': { 'total': 0, 'correct': 0 },
    '3': { 'total': 0, 'correct': 0 },
    '4': { 'total': 0, 'correct': 0 },
    '5': { 'total': 0, 'correct': 0 },
    '6': { 'total': 0, 'correct': 0 },
    '7': { 'total': 0, 'correct': 0 },
    '8': { 'total': 0, 'correct': 0 },
    '9': { 'total': 0, 'correct': 0 },
}
# data = {
#     '0': {'correct': 973, 'total': 980},
#     '1': {'correct': 1126, 'total': 1135},
#     '2': {'correct': 1006, 'total': 1032},
#     '3': {'correct': 995, 'total': 1010},
#     '4': {'correct': 961, 'total': 982},
#     '5': {'correct': 871, 'total': 892},
#     '6': {'correct': 944, 'total': 958},
#     '7': {'correct': 996, 'total': 1028},
#     '8': {'correct': 950, 'total': 974},
#     '9': {'correct': 970, 'total': 1009}
# }

def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    nsamples, nx, ny = trainX.shape
    trainX = trainX.reshape((nsamples, nx*ny ))
    nsamples, nx, ny = testX.shape
    testX = testX.reshape((nsamples, nx*ny))
    return trainX, trainY, testX, testY
    
def process_pixels(train, test):
    test_norm = test.astype('float32')
    test_norm = test_norm / 255.0
    return train, test_norm

def main():
    _, __, test_x, test_y = load_dataset()
    _,test_x = process_pixels(_,test_x)

    total_images = test_x.shape[0]
    print('total_images: ',total_images)

    for i in range(total_images):
        current_image = test_x[i]
        current_label = str(test_y[i])
        result = classifier.predict([current_image])
        result = str(result[0])
        # print('actual:',current_label,', predicted:', result)
        data[current_label]['total'] += 1
        if current_label == result[0]:
            data[current_label]['correct'] += 1

    pprint(data)

    total = [ digit['total'] for digit in data.values()]
    correct = [ digit['correct'] for digit in data.values()]

    # Bar Chart reference: https://benalexkeen.com/bar-charts-in-matplotlib/
    ind = np.arange(len(data.keys())) 
    width = 0.25       
    plt.bar(ind, correct, width, label='correctly predicted')
    plt.bar(ind + width, total, width, label='total images')

    plt.ylabel('Count')
    plt.title('Digits')

    plt.xticks(ind + width / 2, (str(i) for i in range(10)))
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()