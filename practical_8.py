import math

n = int(input("Enter number of records: "))
m = int(input("Enter number of features: "))

names = []
for i in range(m):
    names.append(input("Enter feature name: "))

X = []
Y = []

print("\nEnter dataset values:")
for i in range(n):
    sample = []
    print("\nRecord", i + 1)
    for j in range(m):
        sample.append(float(input(names[j] + ": ")))
    Y.append(int(input("Class (0/1): ")))
    X.append(sample)

w = [0.0] * m
b = 0.0
lr = 0.1
epochs = 1000


def sigmoid(z):
    if z < -500:
        return 0.0
    if z > 500:
        return 1.0
    return 1 / (1 + math.exp(-z))


# Batch gradient descent
for epoch in range(epochs):
    dw = [0.0] * m
    db = 0.0
    for i in range(n):
        z = b
        for j in range(m):
            z += w[j] * X[i][j]
        p = sigmoid(z)
        error = p - Y[i]
        db += error
        for j in range(m):
            dw[j] += error * X[i][j]
    b -= lr * db / n
    for j in range(m):
        w[j] -= lr * dw[j] / n

print("\nCoefficients:")
for j in range(m):
    print(names[j], ":", round(w[j], 4))
print("Intercept:", round(b, 4))

print("\nEnter values for a new sample:")
sample = []
for name in names:
    sample.append(float(input(name + ": ")))

z = b
for j in range(m):
    z += w[j] * sample[j]

probability = sigmoid(z)
predicted_label = 1 if probability >= 0.5 else 0

print("\nProbability:", round(probability, 4))
print("Predicted Class:", predicted_label)