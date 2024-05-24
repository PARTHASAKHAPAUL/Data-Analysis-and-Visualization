# Data-Analysis-and-Visualization(MA5755)

## Gaussian Discriminant Analysis Implementation(2nd Assignment)

In this project, we implement and test Gaussian discriminant analysis, following Sec 9.2 in Murphy’s book. The project is divided into three tasks:

### Task 1: Data Generation
Generate data points for all classes and compute approximations of mean (ˆµc) and covariance (Σˆc) based on the data points. Also, generate a second set of points for testing classification functions.

### Task 2: Evaluate Multivariate Gaussian Distribution
Evaluate the multivariate Gaussian distribution on the test points of all classes to obtain the probabilities \(p(X = t|Y = c)\).

### Task 3: Compute Class Probabilities
Compute the probabilities \(p(Y = c|X = t)\) on the test points using Bayes’ formula and find the class with the maximal probability for each test point.

#### Subroutines

##### For Task 1
```python
def getArtificialData(mean, cov, nx, nt):
    # Generates random vectors for data and test points
    # Returns data points (x), test points (t), mean (mu), and covariance matrix (Sgm)
