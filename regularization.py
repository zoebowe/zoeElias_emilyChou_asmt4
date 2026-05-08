#------------------------------------
# Author: T. D. Devlin 
#-----------------------------------

import math
from math import sin, pi
from random import random


def f(x):
    return sin(pi * x)


def generate_training_examples(n=2):
    xs = [random() * 2 - 1 for _ in range(n)]
    return [(x, f(x)) for x in xs]


def fit_without_reg(examples):
    """Computes values of w0 and w1 that minimize the sum-of-squared-errors cost function

    Args:
    - examples: a list of two (x, y) tuples, where x is the feature and y is the label
    """
    w0 = 0
    w1 = 0
    ## BEGIN YOUR CODE ##

    x1, y1 = examples[0]
    x2, y2 = examples[1]

    w1 = (y2 - y1) / (x2 - x1)
    w0 = y1 - w1 * x1

    ## END YOUR CODE ##
    return w0, w1


def fit_with_reg(examples, lambda_hp):
    """Computes values of w0 and w1 that minimize the regularized sum-of-squared-errors cost function

    Args:
    - examples: a list of two (x, y) tuples, where x is the feature and y is the label
    - lambda_hp: a float representing the value of the lambda hyperparameter; a larger value means more regularization
    """
    w0 = 0
    w1 = 0
    ## BEGIN YOUR CODE ##

    eta = 0.05

    for _ in range(1000):
        d_w0 = 0
        d_w1 = 0

        for x, y in examples:
            prediction = w0 + w1 * x
            error = y - prediction

            d_w0 += -2 * error
            d_w1 += -2 * x * error

        d_w0 += 2 * lambda_hp * w0
        d_w1 += 2 * lambda_hp * w1

        w0 -= eta * d_w0
        w1 -= eta * d_w1

    ## END YOUR CODE ##
    return (w0, w1)


def test_error(w0, w1):
    n = 100
    xs = [i/n for i in range(-n, n + 1)]
    return sum((w0 + w1 * x - f(x)) ** 2 for x in xs) / len(xs)


if __name__ == "__main__": 
    
    ## BEGIN YOUR SIMULATION CODE ##

    num_trials = 1000

    total_error_without_reg = 0
    total_error_with_reg = 0

    for _ in range(num_trials):
        examples = generate_training_examples()

        w0_no_reg, w1_no_reg = fit_without_reg(examples)
        w0_reg, w1_reg = fit_with_reg(examples, 1)

        total_error_without_reg += test_error(w0_no_reg, w1_no_reg)
        total_error_with_reg += test_error(w0_reg, w1_reg)

    avg_error_without_reg = total_error_without_reg / num_trials
    avg_error_with_reg = total_error_with_reg / num_trials

    print("Average test error without regularization:", avg_error_without_reg)
    print("Average test error with regularization:", avg_error_with_reg)