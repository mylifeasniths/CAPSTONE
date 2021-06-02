import joblib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import gzip
from tensorflow.keras.datasets  import mnist
# from tensorflow.keras.utils  import to_categorical

def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    nsamples, nx, ny = trainX.shape
    trainX = trainX.reshape((nsamples, nx*ny ))
    nsamples, nx, ny = testX.shape
    testX = testX.reshape((nsamples, nx*ny))

    # trainY = to_categorical(trainY)
    # testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def process_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize the pixel range from 0-255 to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

def main():
    print('loading dataset...')
    trainX, trainY, testX, testY = load_dataset()
    trainX,testX = process_pixels(trainX,testX)

    print('training model...')
    classifier = SVC()
    classifier.fit(trainX,trainY)

    print('Evaluating model... ')
    
    accuracy = classifier.score(testX,testY)
    print('Accuracy: ',accuracy*100)

    joblib.dump(classifier,'svm_model.gz',compress=('gzip',3))
    print('Modal saved')

main()