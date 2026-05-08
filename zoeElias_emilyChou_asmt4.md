---
title: "CSC 665 Sec 3 Assignment 4"
author: "Zoe Elias, Emily Chou"
date: "April 23, 2026"
geometry: margin=1in
fontsize: 11pt
---

# Problem I

## Margin-Based Linear Classification (Hinge Loss)

### 1. Hinge Loss and Large-Margin Classification

Write answer here.

$$
c(\hat{y}, y) = \max(0, 1 - y\hat{y})
$$

where $\hat{y} = h(x_1,x_2)$.

### 2. Subgradients of the Hinge Loss

The hypothesis is:

$$
h(x_1,x_2)=w_0+w_1x_1+w_2x_2
$$

Define the margin:

$$
m = yh(x_1,x_2)
$$

The hinge loss can be written as:

$$
c(h(x_1,x_2),y)=
\begin{cases}
0, & \text{if } m \ge 1 \\
1-yh(x_1,x_2), & \text{if } m < 1
\end{cases}
$$

### 3. Stochastic Gradient Descent

Initial weights:

$$
(w_0,w_1,w_2)=(0,0,0)
$$

Learning rate:

$$
\eta = 0.1
$$

| Iteration | $(x_1,x_2,y)$ | Margin $m$ | Updated $(w_0,w_1,w_2)$ |
|---:|---:|---:|---:|
| 1 | $(-4,0,-1)$ |  |  |
| 2 | $(-1,1,+1)$ |  |  |
| 3 | $(0,-1,-1)$ |  |  |
| 4 | $(2,1,+1)$ |  |  |
| 5 | $(3,0,+1)$ |  |  |
| 6 | $(6,-1,-1)$ |  |  |

### 4. Misclassification Rate

Final hypothesis:

$$
h(x_1,x_2)=
$$

| Point | True $y$ | $h(x_1,x_2)$ | Prediction | Correct? |
|---:|---:|---:|---:|---:|
| $(-4,0)$ | $-1$ |  |  |  |
| $(-1,1)$ | $+1$ |  |  |  |
| $(0,-1)$ | $-1$ |  |  |  |
| $(2,1)$ | $+1$ |  |  |  |
| $(3,0)$ | $+1$ |  |  |  |
| $(6,-1)$ | $-1$ |  |  |  |

Misclassification rate:

$$
\frac{\text{number of mistakes}}{\text{total examples}} =
$$

# Problem II

## Regularization

### 1. Gradients of the Sum-of-Squared-Errors Cost

Write answer here.

### 2. Gradients with $\ell_2$ Regularization

Write answer here.

### 5. Experiment Results

Write answer here.

### 6. Extra Credit: Visualization

Write answer here.

# GenAI Usage

GenAI was used to help check derivations, verify calculations, and format the final solution. The output was reviewed and edited before submission.
