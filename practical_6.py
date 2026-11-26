n = int(input("Enter number of records: "))

X = []
Y = []

print("\nEnter dataset values:")
for i in range(n):
    X.append(float(input("X: ")))
    Y.append(float(input("Y: ")))

degree = 2
size = degree + 1

# Build the normal equations  A . b = B
A = []
B = []
for i in range(size):
    row = []
    for j in range(size):
        row.append(sum(x ** (i + j) for x in X))
    A.append(row)

for i in range(size):
    B.append(sum((X[k] ** i) * Y[k] for k in range(n)))

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

b0, b1, b2 = B

print("\nCoefficients:")
print("b0:", round(b0, 4))
print("b1:", round(b1, 4))
print("b2:", round(b2, 4))

print("\nPolynomial Equation:")
print("Y =", round(b0, 4), "+", round(b1, 4), "X +", round(b2, 4), "X^2")

x_new = float(input("\nEnter X value for prediction: "))
y_pred = b0 + b1 * x_new + b2 * (x_new ** 2)
print("Predicted Y:", round(y_pred, 4))