# Artificial Neural Network

## Part 1 - installing the deep learning libraries

""" 
Theano is an open source numerical computation Library
very efficient for fast numerical computation
Based on numpy syntax
can run not only on CPU(Central processing Unit) but also on GPU (Graphic processing Unit)
"""


"""
GPU - parallel Computation in neural Network - computations efficiency - 
more powerful - many more cores - able to run a lot more floating points calculations per second than CPU
GPU specialized for highly compute intensive task and parallel computations
"""
"""
involves parallel computations
forward propagating the activations of the different neurons in the neural network 
when the error is back propagated in the neural network.
"""

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

"""
Tensorflow - open source numerical computation Library
under apache 2.0 Licence


"""
# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html


"""
Keras is a machine learning library based on tenserflow and Theaonos
Can build deep learning model with lesser no. of code line

"""
# Installing Keras
# pip install --upgrade keras

## Part 2 - Data Pre-Processing

# Importing the libraries
import numpy as np    #library that contains mathematical tools.
import matplotlib.pyplot as plt   #going to help us plot nice charts.
import pandas as pd 

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #we select all rows and all columns except 1st #independent varaible matrix
y = dataset.iloc[:, 13].values # we select all rows, 3rd column

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## Part 3 - Make ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # required to initialize our neural network
from keras.layers import Dense # required to build the layers of our neural network

# Intializing ANN
classifier = Sequential()

# Adding input layer and first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))
"""
avg of the number of nodes in the input layer and a number of nodes in the output layer.
randomly initialized the weights as small numbers close to zero . 
we can randomly initialised them with a uniform function - which will initialize the weights
according to a uniform distribution.
it will make sure that the weights are small numbers close to zero.
activation = rectfier activation function
input layer has 11 neurons
"""
# Adding 2nd hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Add output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# activation function for o/p layer is sigmoid function

# Compiling Ann
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
"""
# optimizer is the alogorithm to update weights - Stocastic Gradient - ADAM 
# loss function - logarithimic loss function - binary_cross entropy - binary o/p
categorical_cross entropy - categorical variables
"""

# Fitting Ann to training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


#Predict Test set result
y_pred = classifier.predict(X_test)
y_pred1 = (y_pred > 0.5)

#Making confusion Matrix
from sklearn.metrics import confusion_matrix #confusion_matrix is func not class
cm = confusion_matrix(y_test,y_pred1)

# Predicting the single overvation
"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""
new_pred = classifier.predict(sc_X.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))) ## [[]] - 2 dimensional array with rows value
new_prediction = (new_pred > 0.5)

## Part 4 - evaluate, improve ann

# Evaluation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
# Builds the architecuture of ann
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()