n = int(input("Enter number of records: "))

X = []
Y = []

print("\nEnter dataset values:")
for i in range(n):
    x = float(input("X: "))
    y = float(input("Y: "))
    X.append(x)
    Y.append(y)

mean_x = sum(X) / n
mean_y = sum(Y) / n

num = 0
den = 0
for i in range(n):
    num += (X[i] - mean_x) * (Y[i] - mean_y)
    den += (X[i] - mean_x) ** 2

if den == 0:
    print("\nAll X values are identical, so the slope cannot be computed.")
else:
    m = num / den
    c = mean_y - m * mean_x

    print("\nSlope (m):", round(m, 4))
    print("Intercept (c):", round(c, 4))
    print("Regression Equation: Y =", round(m, 4), "X +", round(c, 4))

    x_new = float(input("\nEnter X value for prediction: "))
    y_pred = m * x_new + c
    print("Predicted Y:", round(y_pred, 4))