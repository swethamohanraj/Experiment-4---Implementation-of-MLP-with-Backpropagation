# Experiment-4---Implementation-of-MLP-with-Backpropagation

## AIM:
To implement a Multilayer Perceptron for Multi classification

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.
A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.
MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.
 
MLP has the following features:

Ø  Adjusts the synaptic weights based on Error Correction Rule

Ø  Adopts LMS

Ø  possess Backpropagation algorithm for recurrent propagation of error

Ø  Consists of two passes

  	(i)Feed Forward pass
	         (ii)Backward pass
           
Ø  Learning process –backpropagation

Ø  Computationally efficient method

![image 10](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)

3 Distinctive Characteristics of MLP:

Ø  Each neuron in network includes a non-linear activation function

![image](https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png)

Ø  Contains one or more hidden layers with hidden neurons

Ø  Network exhibits high degree of connectivity determined by the synapses of the network

3 Signals involved in MLP are:

 Functional Signal

*input signal

*propagates forward neuron by neuron thro network and emerges at an output signal

*F(x,w) at each neuron as it passes

Error Signal

   *Originates at an output neuron
   
   *Propagates backward through the network neuron
   
   *Involves error dependent function in one way or the other
   
Each hidden neuron or output neuron of MLP is designed to perform two computations:

The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron

The computation of an estimate of the gradient vector is needed for the backward pass through the network

TWO PASSES OF COMPUTATION:

In the forward pass:

•       Synaptic weights remain unaltered

•       Function signal are computed neuron by neuron

•       Function signal of jth neuron is
            ![image](https://user-images.githubusercontent.com/112920679/198814313-2426b3a2-5b8f-489e-af0a-674cc85bd89d.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814328-1a69a3cd-7e02-4829-b773-8338ac8dcd35.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814339-9c9e5c30-ac2d-4f50-910c-9732f83cabe4.png)



If jth neuron is output neuron, the m=mL  and output of j th neuron is
               ![image](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)

Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer
![image](https://user-images.githubusercontent.com/112920679/198814353-276eadb5-116e-4941-b04e-e96befae02ed.png)


In the backward pass,

•       It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

•        it changes the synaptic weight by delta rule

![image](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)



## ALGORITHM:

1.Import the necessary libraries of python.

2. After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3. Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.

4. Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%.
Normalize our dataset. 

5.In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.

6.Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.

7.In order to get the predicted values we call the predict() function on the testing data set.

8. Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.

## PROGRAM 
```python
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('IRIS.csv')
df.head

names = ['sepal-length','sepal-width','petal-length','petal-width','Class']

# Take first 4 columns ans assign them to variable "X"
X = df.iloc[:,0:4]
# Take first 5th columns and assign them to variable "Y". Object dtype refers to strings
Y = df.select_dtypes(include=[object])
X.head()
Y.head()

# Y actually contains all categories or classes
Y.species.unique()

# Now transforming categorial into numerical values
le = preprocessing.LabelEncoder()
Y = Y.apply(le.fit_transform)
Y.head()

# Train and test split (80% of data into training set and 20% into test data)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.20)

# Feature Scaling
scaler = StandardScaler() 
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(X_train,Y_train.values.ravel())
predictions = mlp.predict(X_test)
print(predictions)

# Evaluation of algorithm performance in classifying flowers
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
```

## OUTPUT 
df.head():
![image](https://user-images.githubusercontent.com/94228215/231656995-7685aa8a-4b3a-4ec6-8d91-afebbde5323a.png)
X.head():
![image](https://user-images.githubusercontent.com/94228215/231657038-9af2a39c-7f14-4a00-ac9b-c5835296d956.png)
Y.head():
![image](https://user-images.githubusercontent.com/94228215/231657134-35c629de-d539-459e-9a41-b7064a830933.png)
Unique Values in Y:
![image](https://user-images.githubusercontent.com/94228215/231657192-727c7437-ab23-4136-8664-6f797323ce8c.png)
Transforming Categorical to numerical values:
![image](https://user-images.githubusercontent.com/94228215/231657232-90541147-ba71-40cf-a07d-5bb7034bf5e3.png)
Predictions:
![image](https://user-images.githubusercontent.com/94228215/231657273-2506b0df-73d7-48b7-8ce7-762d9316ddbf.png)
Confusion Matrix:
![image](https://user-images.githubusercontent.com/94228215/231657349-8385d2d5-4113-4e81-8cf2-159e13e33ddb.png)
Classification_report:
![image](https://user-images.githubusercontent.com/94228215/231657406-6c0039a6-c6d6-43fe-9a95-df6cd266d4a0.png)



## RESULT
Thus, a program to implement Multilayer Perceptron for Multi Classification is successfully created and executed.
