import joblib
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error
from predict import load_test_dataset, pre_processing

MODAL_NAME  = 'operators_svm_model.gz'
classifier = joblib.load(MODAL_NAME)
data = {
    'add': { 'total': 0, 'correct': 0 },
    'sub': { 'total': 0, 'correct': 0 },
    'mul': { 'total': 0, 'correct': 0 },
    'div': { 'total': 0, 'correct': 0 },
}

def print_barchart():
    total = [ digit['total'] for digit in data.values()]
    correct = [ digit['correct'] for digit in data.values()]
    # Bar Chart reference: https://benalexkeen.com/bar-charts-in-matplotlib/
    ind = np.arange(len(data.keys())) 
    width = 0.35       
    plt.bar(ind, correct, width, label='correctly predicted images')
    plt.bar(ind + width, total, width, label='total images')

    plt.ylabel('Count')
    plt.xlabel('Operators')
    # plt.title('Operators')

    plt.xticks(ind + width / 2, ('add', 'sub', 'mul', 'div'))
    plt.legend(loc='best')
    plt.show()

def main():
    test_x, test_y = load_test_dataset()
    test_x = pre_processing(test_x)

    total_images = test_x.shape[0]

    for i in range(total_images):
        current_image = test_x[i]
        current_label = test_y[i]
        result = classifier.predict([current_image])
        result = result[0]

        data[current_label]['total'] += 1
        if current_label == result:
            data[current_label]['correct'] += 1

    pprint(data)


if __name__ == "__main__":
    main()
    print_barchart()
