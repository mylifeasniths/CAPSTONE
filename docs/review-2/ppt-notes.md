# CNN (Convonutional Neural Network)
- A convolutional neural network is a subset of deep neural networks, most commonly applied to analyze two dimentional data(Image and Video).

## Convolutional Layer
- This layer uses a filter (also called as kernel) which is an array of weights to extract features from the input image. One layer can have many filters
- Conv2D class from tensorflow is used to create an instance of Convlutional layer
- _add a code implementation image_

## Pooling Layer
- This layer reduces the dimensions of the data coming from Convolutional layer which in return reduces the computations, number of parameters, reduces over fitting and therefore making the entire process much faster.
- MaxPoling has been used
- MaxPooling2D class from tensorflow is used to create an instance of Pooling layer
- _< add a code implementation image >_

## Activation function
- Used in the hidden layers of a neural network. This allows the model to learn more complex functions than a network trained using a linear activation function. Two activation functions are used
- ReLU (Rectified Linear Unit)
    * Conv2D instance provides an easy way of adding ReLU function 
    * _< add a code implementation image >_

- Softmax
    * Commonly used in Output Layer for Multiclass Classification
    * It returns a Probablity distrubution of target classes for a classification problem
    * Conv2D instance provides an easy way of adding Softmax function 
    * _< add a code implementation image >_

## Accuracy
- 99.0
- _< add a output image >_

-------------------------------------------------------------------------------------

# SVM (Support Vector Machine)
- Popoular Supervised learning algorithms, commonly used for classification

### Code Implementation
- SVC(Support Vector Classifier) is a Class from sklearn package which is been used to create a classifier (modal) instance.
- _< add a code implementation image >_

### Accuracy
- 97.92
- _< add a output image >_
