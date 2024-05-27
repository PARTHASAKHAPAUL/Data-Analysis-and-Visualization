# Data-Analysis-and-Visualization(MA5755)

## **1. `Gaussian Discriminant Analysis Implementation`**(**`2nd Assignment`**)

In this project, we implement and test Gaussian discriminant analysis, following Sec 9.2 in Murphy’s book. The project is divided into three tasks:

### Task 1: Data Generation
Generate data points for all classes and compute approximations of mean (ˆµc) and covariance (Σˆc) based on the data points. Also, generate a second set of points for testing classification functions.
```python
def getArtificialData(mean, cov, nx, nt):
    # Generates random vectors for data and test points
    # Returns data points (x), test points (t), mean (mu), and covariance matrix (Sgm)
```

### Task 2: Evaluate Multivariate Gaussian Distribution
Evaluate the multivariate Gaussian distribution on the test points of all classes to obtain the probabilities \(p(X = t|Y = c)\).
```python
def evaluateMultiVarGauss(t, mu, Sgm):
    # Evaluates multivariate Gaussian distribution on test points
    # Returns probability density function (p)
```

### Task 3: Compute Class Probabilities
Compute the probabilities \(p(Y = c|X = t)\) on the test points using Bayes’ formula and find the class with the maximal probability for each test point.
```python
def calculateTestSet(t, pXY, yEx):
    # Computes probabilities and labels for each test point
    # Prints probabilities, actual and computed labels
    # Counts mislabeled points
```

### Experimentation
#### Artificial Data
- Generate three classes with specified parameters.
- Test the quadratic decision boundaries.
- Modify the code to test tied covariances (linear decision boundaries) and compare results.

#### Iris Dataset
- Import the iris dataset.
- Extract petal/sepal dimensions and corresponding labels.
- Implement getIrisData() to prepare data similar to getArtificialData().
- Test the methodology on the iris dataset and count misclassified items in the test set.

This project(`2nd Assignment`) provides an overview of the `Gaussian Discriminant Analysis` implementation and outlines the tasks involved along with the experimentation process.





## **2. `Logistic Regression for Stock Market Prediction`**(**`Assignment-3: A3`**)

### Introduction
This project implements binary logistic regression to predict the direction of the stock market using historical data. Two gradient descent methods, full gradient and stochastic gradient descent, are employed to determine the optimal parameters for the logistic regression model.

### Part 1: Implementation of Gradient Descent Methods
In this part, we implemented the following functions:
- `fullGradient(w, X, y)`: Returns the gradient for logistic regression.
- `randGradient(w, X, y, n)`: Returns the stochastic gradient for logistic regression.
- `fdescent(X, y, rho, job, nEpoch)`: Performs gradient descent using either full or stochastic gradient method.

### Part 2: Testing the Method
#### Data Understanding and Preparation
- Computed the correlation matrix to understand the relationships between variables.
- Split the [Smarket data](https://github.com/PARTHASAKHAPAUL/Data-Analysis-and-Visualization/blob/main/smarket.csv) into training set (2001-2004) and test set (2005).

#### Experimentation and Evaluation
- Ran logistic regression using both full and stochastic descent methods.
- Plotted the histories of the objective function for comparison.
- Evaluated which method provides better results and consumes fewer floating-point operations.

#### Classification and Misclassification Analysis
- Created a table containing probabilities, classifiers, and actual directions for each test point.
- Analyzed the misclassified points to assess the performance of the model.

#### Prediction for the Following Day
- Modified the [dataset-Smarket_modified](https://github.com/PARTHASAKHAPAUL/Data-Analysis-and-Visualization/blob/main/smarket_modified.xlsx) to predict the direction of the stock market for the following day.
- Re-ran the classification to assess the misclassification rate with the modified dataset.

### Conclusion
The project demonstrates the application of logistic regression and gradient descent methods for stock market prediction. While the model shows promising results, it's important to note the limitations, particularly regarding the availability of data for real-time prediction.


## **3. `Building a Neural Network from Scratch for Digit Recognition`** (**`Assignment-4: A4`**)

### Introduction
This project involves building a digit recognizer neural network from scratch using numpy, pandas, and matplotlib. The goal is to replicate and understand the neural network implementation as demonstrated by [Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU) and subsequently enhance the network to achieve higher accuracy.

### Part 1: Understanding and Reproducing the Base Model
#### Steps:
1. **Watch the Tutorial**:
   - Watch the YouTube video by Samson Zhang to understand the implementation.
   - Review the code available on the Kaggle website linked in the video description.

2. **Code Understanding and Adjustments**:
   - Ensure you understand each line of code as explained in the video.
   - Adjust the code to load the MNIST data from the keras datasets in Google Colab.
   - Scale the grayscale values of the images to the interval [0, 1].

3. **Verify Initial Model**:
   - Run the initial model with 500 forward/backward propagation steps and a learning rate of α = 0.1.
   - Ensure the accuracy of the trained network is between 82% and 85%.

### Part 2: Enhancing the Neural Network
#### Steps:
1. **Network Enlargement**:
   - Add an additional hidden layer to the neural network.
   - Use 20 nodes in the first hidden layer and 10 nodes in the second hidden layer.

2. **Training the Enhanced Model**:
   - Run the propagation with 1000 steps and a learning rate of α = 0.1.
   - Measure the accuracy rate of the enhanced network.

### Part 3: Implementation Details
#### Base Model Implementation
- **Loading Data**:
```python
# Importing the necessary library to load the MNIST dataset
# Loading the MNIST dataset and storing the training and testing data along with their respective labels
# Reshaping the images to a 2D array and normalizing pixel values to the range [0, 1]
# Encoding the labels into one-hot format
```

### Model Training:
```python
# Initialization of weights and biases
# Forward and backward propagation
# Training loop for 500 steps with learning rate α = 0.1
```

### Enhanced Model Implementation
#### Enhanced Network Architecture:
```python
# Define neural network with additional hidden layer
# 20 nodes in the first hidden layer, 10 nodes in the second hidden layer
```
#### Training the Enhanced Model:
```python
# Training loop for 1000 steps with learning rate α = 0.1
```

### Results and Analysis
#### Accuracy of Base Model:
- Document the accuracy rate of the initial model (should be between 82% and 85%).

#### Accuracy of Enhanced Model:
- Document the accuracy rate of the enhanced model after 1000 steps.

### Conclusion
The project demonstrates the implementation and enhancement of a neural network for digit recognition using the MNIST dataset. By adding additional hidden layers and nodes, the model's accuracy is expected to improve. The results and accuracy rates for both the base and enhanced models are documented for comparison.



