n = int(input("Enter number of records: "))
m = int(input("Enter number of features: "))

names = []
for i in range(m):
    names.append(input("Enter feature name: "))
target_field = input("Enter target variable name: ")

X = []
Y = []

print("\nEnter dataset values:")
for i in range(n):
    sample = [1]                      # leading 1 is for the intercept b0
    print("\nRecord", i + 1)
    for j in range(m):
        sample.append(float(input(names[j] + ": ")))
    Y.append(float(input(target_field + ": ")))
    X.append(sample)

size = m + 1

# Normal equations: (X^T X) b = X^T Y
A = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(size)]
     for i in range(size)]
B = [sum(X[k][i] * Y[k] for k in range(n)) for i in range(size)]

# Gauss-Jordan elimination (with row swap if the pivot is zero)
for i in range(size):
    if A[i][i] == 0:
        for r in range(i + 1, size):
            if A[r][i] != 0:
                A[i], A[r] = A[r], A[i]
                B[i], B[r] = B[r], B[i]
                break
    pivot = A[i][i]
    for j in range(i, size):
        A[i][j] /= pivot
    B[i] /= pivot
    for k in range(size):
        if k != i:
            factor = A[k][i]
            for j in range(i, size):
                A[k][j] -= factor * A[i][j]
            B[k] -= factor * B[i]

print("\nCoefficients:")
for i in range(size):
    print("b" + str(i), ":", round(B[i], 4))

print("\nRegression Equation:")
equation = "Y = " + str(round(B[0], 4))
for i in range(1, size):
    equation += " + " + str(round(B[i], 4)) + " " + names[i - 1]
print(equation)

sample = [1]
print("\nEnter values for a new sample:")
for name in names:
    sample.append(float(input(name + ": ")))

y_pred = sum(B[i] * sample[i] for i in range(size))
print("Predicted Y:", round(y_pred, 4))