# Data-Analysis-and-Visualization(MA5755)

## Gaussian Discriminant Analysis Implementation(**`2nd Assignment`**)

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
