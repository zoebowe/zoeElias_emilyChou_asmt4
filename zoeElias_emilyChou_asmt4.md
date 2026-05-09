---
title: "CSC X65 Sec 3 Assignment 4"
author: "Zoe Elias, Emily Chou"
geometry: margin=1in
fontsize: 11pt
---

# Problem I

## Margin-Based Linear Classification (Hinge Loss)

### 1. Hinge Loss and Large-Margin Classification

The hinge loss for binary classification with labels $y \in \{-1, +1\}$ is defined as:

$$
c(\hat{y}, y) = \max\!\big(0,\, 1 - y\hat{y}\big)
$$

where

$$
\hat{y} = h(x_1, x_2).
$$

Hinge loss is widely used in large-margin classifiers, particularly Support Vector Machines (SVMs), where the goal is to find a decision boundary that separates classes with the largest possible margin. According to standard machine learning literature, hinge loss penalizes both misclassified points and correctly classified points that lie within the margin boundary (Wikipedia, Hinge Loss). This differs from simple classification loss functions that only penalize incorrect predictions.

From a theoretical perspective, hinge loss is designed to approximate the 0--1 loss while remaining convex, which allows it to be efficiently optimized using gradient-based methods (Scikit-learn Documentation). Additionally, large-margin methods are known to improve generalization performance because they produce classifiers that are less sensitive to noise in the training data (Cornell CS4780 Lecture Notes).

Therefore, hinge loss is a sensible objective for large-margin classification because it explicitly encourages correct predictions with high confidence, increases robustness to noise, and provides a tractable optimization objective.

**Sources (Clickable Links)**

- [Hinge Loss - Wikipedia](https://en.wikipedia.org/wiki/Hinge_loss)  
- [Hinge Loss - Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html)  
- [Cornell CS4780 Lecture Notes](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote09.html)  

\newpage

### 2. Subgradients of the Hinge Loss

The hypothesis is:

$$
h(x_1, x_2) = w_0 + w_1 x_1 + w_2 x_2
$$

The hinge loss is:

$$
c(h(x_1, x_2), y) = \max\big(0,\, 1 - y h(x_1, x_2)\big)
$$

Define the margin:

$$
m = y h(x_1, x_2)
$$

So:

$$
c(h(x_1, x_2), y) =
\begin{cases}
0, & \text{if } m \ge 1 \\
1 - y h(x_1, x_2), & \text{if } m < 1
\end{cases}
$$

Substitute the hypothesis:

$$
c(h(x_1, x_2), y) =
\begin{cases}
0, & \text{if } m \ge 1 \\
1 - y\big(w_0 + w_1 x_1 + w_2 x_2\big), & \text{if } m < 1
\end{cases}
$$

**Case 1: $m \ge 1$**

If:

$$
m = y h(x_1, x_2) \ge 1
$$

then:

$$
c(h(x_1, x_2), y) = 0
$$

Since 0 is constant:

$$
\frac{\partial}{\partial w_0}(0) = 0, \quad
\frac{\partial}{\partial w_1}(0) = 0, \quad
\frac{\partial}{\partial w_2}(0) = 0
$$

So:

$$
\frac{\partial c}{\partial w_0} = 0, \quad
\frac{\partial c}{\partial w_1} = 0, \quad
\frac{\partial c}{\partial w_2} = 0
$$

**Case 2: $m < 1$**

If:

$$
m = y h(x_1, x_2) < 1
$$

then:

$$
c(h(x_1, x_2), y) = 1 - y\big(w_0 + w_1 x_1 + w_2 x_2\big)
$$

Distribute $-y$:

$$
c = 1 - y w_0 - y w_1 x_1 - y w_2 x_2
$$

**Derivative with respect to $w_0$**

$$
\frac{\partial c}{\partial w_0}
=
\frac{\partial}{\partial w_0}
\left(1 - y w_0 - y w_1 x_1 - y w_2 x_2\right)
$$

$$
=
\frac{\partial}{\partial w_0}(1)
-
\frac{\partial}{\partial w_0}(y w_0)
-
\frac{\partial}{\partial w_0}(y w_1 x_1)
-
\frac{\partial}{\partial w_0}(y w_2 x_2)
$$

$$
= 0 - y - 0 - 0 = -y
$$

**Derivative with respect to $w_1$**

$$
\frac{\partial c}{\partial w_1}
=
\frac{\partial}{\partial w_1}
\left(1 - y w_0 - y w_1 x_1 - y w_2 x_2\right)
$$

$$
=
\frac{\partial}{\partial w_1}(1)
-
\frac{\partial}{\partial w_1}(y w_0)
-
\frac{\partial}{\partial w_1}(y w_1 x_1)
-
\frac{\partial}{\partial w_1}(y w_2 x_2)
$$

$$
= 0 - 0 - y x_1 - 0 = -y x_1
$$

**Derivative with respect to $w_2$**

$$
\frac{\partial c}{\partial w_2}
=
\frac{\partial}{\partial w_2}
\left(1 - y w_0 - y w_1 x_1 - y w_2 x_2\right)
$$

$$
=
\frac{\partial}{\partial w_2}(1)
-
\frac{\partial}{\partial w_2}(y w_0)
-
\frac{\partial}{\partial w_2}(y w_1 x_1)
-
\frac{\partial}{\partial w_2}(y w_2 x_2)
$$

$$
= 0 - 0 - 0 - y x_2 = -y x_2
$$

**Final Answer**

$$
\frac{\partial c}{\partial w_0} =
\begin{cases}
0, & \text{if } m \ge 1 \\
- y, & \text{if } m < 1
\end{cases}
$$

$$
\frac{\partial c}{\partial w_1} =
\begin{cases}
0, & \text{if } m \ge 1 \\
- y x_1, & \text{if } m < 1
\end{cases}
$$

$$
\frac{\partial c}{\partial w_2} =
\begin{cases}
0, & \text{if } m \ge 1 \\
- y x_2, & \text{if } m < 1
\end{cases}
$$

Equivalently:

$$
\nabla c =
\begin{cases}
(0, 0, 0), & \text{if } m \ge 1 \\
(-y, -y x_1, -y x_2), & \text{if } m < 1
\end{cases}
$$

At $m = 1$, the hinge loss is not differentiable. A valid subgradient can be any convex combination of $(0,0,0)$ and $(-y, -y x_1, -y x_2)$. For this solution, we use the zero subgradient for the $m \ge 1$ case.

\newpage

### 3. Stochastic Gradient Descent (SGD)

Initial weights:

$$
(w_0, w_1, w_2) = (0, 0, 0)
$$

Learning rate:

$$
\eta = 0.1
$$

Update rule:

$$
w \leftarrow w - \eta \nabla c
$$

From Problem 2, when $m < 1$:

$$
\nabla c = (-y, -y x_1, -y x_2)
$$

Thus:

$$
w \leftarrow w + \eta (y, y x_1, y x_2)
$$

If $m \ge 1$, no update is performed since the gradient is zero.

**Detailed Computation**

**Iteration 1**

$$
(x_1, x_2, y) = (-4, 0, -1)
$$

$$
h(x) = 0
$$

$$
m = (-1)(0) = 0 < 1
$$

$$
w_0 = 0 + 0.1(-1) = -0.1
$$

$$
w_1 = 0 + 0.1(-1)(-4) = 0.4
$$

$$
w_2 = 0 + 0.1(-1)(0) = 0
$$

**Iteration 2**

$$
h(x) = -0.1 + 0.4(-1) + 0(1) = -0.5
$$

$$
m = (-0.5) < 1
$$

$$
(w_0, w_1, w_2) = (0, 0.3, 0.1)
$$

**Iteration 3**

$$
h(x) = 0 + 0.3(0) + 0.1(-1) = -0.1
$$

$$
m = (-1)(-0.1) = 0.1 < 1
$$

$$
(w_0, w_1, w_2) = (-0.1, 0.3, 0.2)
$$

**Iteration 4**

$$
h(x) = -0.1 + 0.3(2) + 0.2(1) = 0.7
$$

$$
m = 0.7 < 1
$$

$$
(w_0, w_1, w_2) = (0, 0.5, 0.3)
$$

**Iteration 5**

$$
h(x) = 0 + 0.5(3) = 1.5
$$

$$
m = 1.5 \ge 1
$$

No update.

**Iteration 6**

$$
h(x) = 0 + 0.5(6) + 0.3(-1) = 2.7
$$

$$
m = (-1)(2.7) = -2.7 < 1
$$

$$
(w_0, w_1, w_2) = (-0.1, -0.1, 0.4)
$$

**Summary Table**

| Iteration | $(x_1,x_2,y)$ | Margin $m$ | Updated $(w_0,w_1,w_2)$ |
|---:|---:|---:|---:|
| 1 | $(-4,0,-1)$ | $0$ | $(-0.1, 0.4, 0)$ |
| 2 | $(-1,1,+1)$ | $-0.5$ | $(0, 0.3, 0.1)$ |
| 3 | $(0,-1,-1)$ | $0.1$ | $(-0.1, 0.3, 0.2)$ |
| 4 | $(2,1,+1)$ | $0.7$ | $(0, 0.5, 0.3)$ |
| 5 | $(3,0,+1)$ | $1.5$ | $(0, 0.5, 0.3)$ |
| 6 | $(6,-1,-1)$ | $-2.7$ | $(-0.1, -0.1, 0.4)$ |

**Final Weights**

$$
(w_0, w_1, w_2) = (-0.1, -0.1, 0.4)
$$

If $m \ge 1$, no update is performed since the gradient is zero.

\newpage

### 4. Misclassification Rate

From Problem 3, the final weights are:

$$
(w_0, w_1, w_2) = (-0.1, -0.1, 0.4)
$$

Therefore, the final hypothesis is:

$$
h(x_1, x_2) = w_0 + w_1x_1 + w_2x_2
$$

Substitute the final weights:

$$
h(x_1, x_2) = -0.1 - 0.1x_1 + 0.4x_2
$$

The prediction rule is:

$$
\hat{y} =
\begin{cases}
+1, & \text{if } h(x_1, x_2) \ge 0 \\
-1, & \text{if } h(x_1, x_2) < 0
\end{cases}
$$

**Detailed Computation**

**Point 1: $(-4, 0)$**

$$
h(-4, 0) = -0.1 - 0.1(-4) + 0.4(0)
$$

$$
= -0.1 + 0.4 + 0 = 0.3
$$

Since $0.3 \ge 0$, $\hat{y} = +1$.  
True label: $y = -1$ → misclassified.

**Point 2: $(-1, 1)$**

$$
h(-1, 1) = -0.1 - 0.1(-1) + 0.4(1)
$$

$$
= -0.1 + 0.1 + 0.4 = 0.4
$$

Since $0.4 \ge 0$, $\hat{y} = +1$.  
True label: $y = +1$ → correct.

**Point 3: $(0, -1)$**

$$
h(0, -1) = -0.1 - 0.1(0) + 0.4(-1)
$$

$$
= -0.1 + 0 - 0.4 = -0.5
$$

Since $-0.5 < 0$, $\hat{y} = -1$.  
True label: $y = -1$ → correct.

**Point 4: $(2, 1)$**

$$
h(2, 1) = -0.1 - 0.1(2) + 0.4(1)
$$

$$
= -0.1 - 0.2 + 0.4 = 0.1
$$

Since $0.1 \ge 0$, $\hat{y} = +1$.  
True label: $y = +1$ → correct.

**Point 5: $(3, 0)$**

$$
h(3, 0) = -0.1 - 0.1(3) + 0.4(0)
$$

$$
= -0.1 - 0.3 + 0 = -0.4
$$

Since $-0.4 < 0$, $\hat{y} = -1$.  
True label: $y = +1$ → misclassified.

**Point 6: $(6, -1)$**

$$
h(6, -1) = -0.1 - 0.1(6) + 0.4(-1)
$$

$$
= -0.1 - 0.6 - 0.4 = -1.1
$$

Since $-1.1 < 0$, $\hat{y} = -1$.  
True label: $y = -1$ → correct.

**Summary Table**

| Point | True $y$ | $h(x_1, x_2)$ | Prediction $\hat{y}$ | Correct? |
|:---:|:---:|:---:|:---:|:---:|
| $(-4, 0)$ | $-1$ | $0.3$ | $+1$ | No |
| $(-1, 1)$ | $+1$ | $0.4$ | $+1$ | Yes |
| $(0, -1)$ | $-1$ | $-0.5$ | $-1$ | Yes |
| $(2, 1)$ | $+1$ | $0.1$ | $+1$ | Yes |
| $(3, 0)$ | $+1$ | $-0.4$ | $-1$ | No |
| $(6, -1)$ | $-1$ | $-1.1$ | $-1$ | Yes |

There are 2 misclassified points out of 6 total points.

Therefore:

$$
\text{Misclassification Rate}
=
\frac{\text{number of misclassified points}}{\text{total number of points}}
$$

$$
=
\frac{2}{6}
=
\frac{1}{3}
\approx 0.3333
$$

$$
\text{Misclassification Rate} = 33.33\%
$$

\newpage

### 5. Extra Credit: Gradient Descent (Full-Batch)

We implement non-stochastic (full-batch) gradient descent, where each update uses the entire dataset rather than a single example.

The update rule is:

$$
w \leftarrow w - \eta \nabla C(w)
$$

where the gradient is computed by summing the hinge loss subgradients over all training examples.

Using a learning rate of:

$$
\eta = 0.1
$$

and stopping when the maximum change in weights is less than:

$$
10^{-5}
$$

the algorithm converges after 59 iterations.

**Final Weights**

$$
(w_0, w_1, w_2) \approx (-0.4, 0.5, 4.0)
$$

Thus, the learned hypothesis is:

$$
h(x_1, x_2) = -0.4 + 0.5x_1 + 4.0x_2
$$

**Training Error**

All points are classified correctly, so:

$$
\text{Training Error} = 0.0
$$

**Comparison with SGD**

From Problem 3, stochastic gradient descent (SGD) produced:

$$
(w_0, w_1, w_2) = (-0.1, -0.1, 0.4)
$$

with training error:

$$
\frac{2}{6} = \frac{1}{3} \approx 0.3333
$$

**Discussion**

Full-batch gradient descent achieves zero training error and finds a decision boundary that perfectly separates the data. In contrast, SGD produces a solution with non-zero training error.

This difference occurs because full-batch gradient descent uses the entire dataset at each update, while the SGD calculation in Problem 3 used only one pass through the data. SGD, on the other hand, updates parameters using only one example at a time, which introduces noise into the optimization process and may prevent convergence to the optimal solution within a single pass through the data.

Additionally, the final weights from gradient descent are significantly larger in magnitude, reflecting a stronger separation between classes. This demonstrates how full-batch optimization can lead to a more accurate and stable model for this dataset.

\newpage

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
