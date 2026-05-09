#------------------------------------
# Author: Zoe Elias, Emily Chou
#------------------------------------

def hinge_gradient(weights, data):
    grad = [0.0, 0.0, 0.0]

    for x1, x2, y in data:
        w0, w1, w2 = weights
        h = w0 + w1 * x1 + w2 * x2
        margin = y * h

        if margin < 1:
            grad[0] += -y
            grad[1] += -y * x1
            grad[2] += -y * x2

    return grad


def training_error(weights, data):
    mistakes = 0

    for x1, x2, y in data:
        w0, w1, w2 = weights
        h = w0 + w1 * x1 + w2 * x2
        prediction = 1 if h >= 0 else -1

        if prediction != y:
            mistakes += 1

    return mistakes / len(data)


def main():
    data = [
        (-4, 0, -1),
        (-1, 1, 1),
        (0, -1, -1),
        (2, 1, 1),
        (3, 0, 1),
        (6, -1, -1),
    ]

    eta = 0.1
    tolerance = 1e-5
    max_iterations = 100000

    weights = [0.0, 0.0, 0.0]

    for iteration in range(max_iterations):
        grad = hinge_gradient(weights, data)

        new_weights = [
            weights[0] - eta * grad[0],
            weights[1] - eta * grad[1],
            weights[2] - eta * grad[2],
        ]

        change = max(abs(new_weights[i] - weights[i]) for i in range(3))
        weights = new_weights

        if change < tolerance:
            break

    print("Gradient Descent Results")
    print("------------------------")
    print("Iterations:", iteration + 1)
    print("Final weights:", weights)
    print(
        f"Learned h(x1, x2) = {weights[0]} + {weights[1]}x1 + {weights[2]}x2"
    )
    print("Training error:", training_error(weights, data))

    print()
    print("SGD Comparison")
    print("--------------")
    print("SGD final weights: [-0.1, -0.1, 0.4]")
    print("SGD learned h(x1, x2) = -0.1 + -0.1x1 + 0.4x2")
    print("SGD training error:", 2 / 6)


if __name__ == "__main__":
    main()
