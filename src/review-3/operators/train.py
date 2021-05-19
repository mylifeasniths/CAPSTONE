from sklearn.svm import SVC
import joblib
import gzip
from dataset import load_dataset

# Idea
# 1. load datasets
# 2. process pixels
# 3. intialize & train model
# 4. evaluate model
# 5. save model

TRAIN_X = 'train_x.npz'
TRAIN_Y = 'train_y.npz'
TEST_X  = 'test_x.npz'
TEST_Y  = 'test_y.npz'
MODAL_NAME  = 'operators_svm_model.gz'

def load_all_datasets():
    """ 
    load and returns training, testing dataset and labels 
    returns : train_x, train_y, test_x, test_y (all numpy arrays)
    """
    train_x = load_dataset(TRAIN_X)
    train_y = load_dataset(TRAIN_Y)
    test_x = load_dataset(TEST_X)
    test_y = load_dataset(TEST_Y)
    return train_x, train_y, test_x, test_y

def pre_processing(train_x,test_x):
    """ 
    process train_x, test_x by convering int to float & normalizing pixels values to the range on 0-1 
    parameters : train_x, test_x (both are numpy arrays)
    returns    : train_norm, test_norm (both are numpy arrays)
    """
    train_norm = train_x.astype('float32')
    test_norm = test_x.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

def main():
    train_x, train_y, test_x, test_y = load_all_datasets()
    train_x, test_x = pre_processing(train_x,test_x)

    print('training model...')
    classifier = SVC()
    classifier.fit(train_x,train_y)

    print('evaluating model... ')
    accuracy = classifier.score(test_x,test_y)
    print('accuracy: ',accuracy*100)    

    joblib.dump(classifier,MODAL_NAME,compress=('gzip',3))
    print(f'modal saved as {MODAL_NAME}')

if __name__ == "__main__" :
    main()