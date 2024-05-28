import numpy as np

def rosenbrock_patrick(X, a=1, b=100):
    X = np.asarray(X)
    X_shifted = X[1:]
    X_minus_shifted = X[:-1]
    return np.sum(a * (X_shifted - X_minus_shifted**2)**2 + b * (X_minus_shifted - 1)**2)


def rosenbrock_np(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rosenbrock_sum(x, a=1, b=100):
    """
    Computes the Rosenbrock function for a given list of coordinates.
    
    Parameters:
    x (list or array): List or array of coordinates.
    a (float): The parameter a in the Rosenbrock function. Default is 1.
    b (float): The parameter b in the Rosenbrock function. Default is 100.
    
    Returns:
    float: The value of the Rosenbrock function.
    """
    return sum(b * (x[i+1] - x[i]**2)**2 + (a - x[i])**2 for i in range(len(x) - 1))


def rosenbrock_old(X, a=1, b=100):
    return sum(a * (X[i+1] - X[i]**2)**2 + b * (X[i] - 1)**2 for i in range(len(X)-1))


def rosenbrock_default(X, a=1, b=100):
    x, y = X
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_new(x, a=1, b=100):  
    return sum((a-x[i])**2 + b*(x[i+1]-x[i]**2)**2 for i in range(len(x)-1))

x = np.random.uniform(size=2)
x = np.array([5, 1])
# Example usage
print(x)
print(f"rosenbrock_new: {rosenbrock_new(x)}")
print(f"rosenbrock_default: {rosenbrock_default(x)}")
print(f"rosenbrock_patrick: {rosenbrock_patrick(x)}")
print(f"rosenbrock_np: {rosenbrock_np(x)}")
print(f"rosenbrock_sum: {rosenbrock_sum(x)}")
print(f"rosenbrock_old: {rosenbrock_old(x)}")
