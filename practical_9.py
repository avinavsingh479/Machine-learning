import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


print("Enter input values:")
x1 = float(input("x1: "))
x2 = float(input("x2: "))
x3 = float(input("x3: "))

target = float(input("Target Output: "))
lr = float(input("Learning Rate: "))

print("\nEnter input-hidden weights:")
wx1h1 = float(input("x1 -> h1: "))
wx1h2 = float(input("x1 -> h2: "))
wx2h1 = float(input("x2 -> h1: "))
wx2h2 = float(input("x2 -> h2: "))
wx3h1 = float(input("x3 -> h1: "))
wx3h2 = float(input("x3 -> h2: "))

print("\nEnter hidden biases:")
b1 = float(input("b1 (h1): "))
b2 = float(input("b2 (h2): "))

print("\nEnter output weights:")
wh1o = float(input("h1 -> output: "))
wh2o = float(input("h2 -> output: "))
bout = float(input("bout: "))

print("\n========== FORWARD PROPAGATION ==========")

net_h1 = x1 * wx1h1 + x2 * wx2h1 + x3 * wx3h1 + b1
net_h2 = x1 * wx1h2 + x2 * wx2h2 + x3 * wx3h2 + b2

h1 = sigmoid(net_h1)
h2 = sigmoid(net_h2)

print("Net input of h1:", round(net_h1, 8))
print("Output of h1:", round(h1, 8))
print("Net input of h2:", round(net_h2, 8))
print("Output of h2:", round(h2, 8))

net_out = h1 * wh1o + h2 * wh2o + bout
out = sigmoid(net_out)

print("Net input of output:", round(net_out, 8))
print("Actual Output:", round(out, 8))
print("Target Output:", round(target, 8))

error = target - out

print("\n========== ERROR ==========")
print("Error:", round(error, 8))

delta_out = error * out * (1 - out)

# hidden deltas use the OLD output weights
delta_h1 = h1 * (1 - h1) * delta_out * wh1o
delta_h2 = h2 * (1 - h2) * delta_out * wh2o

print("\n========== BACKPROPAGATION ==========")
print("Delta Output:", round(delta_out, 8))
print("Delta h1:", round(delta_h1, 8))
print("Delta h2:", round(delta_h2, 8))

print("\n========== UPDATES ==========")

wh1o_new = wh1o + lr * delta_out * h1
wh2o_new = wh2o + lr * delta_out * h2
bout_new = bout + lr * delta_out

wx1h1_new = wx1h1 + lr * delta_h1 * x1
wx2h1_new = wx2h1 + lr * delta_h1 * x2
wx3h1_new = wx3h1 + lr * delta_h1 * x3

wx1h2_new = wx1h2 + lr * delta_h2 * x1
wx2h2_new = wx2h2 + lr * delta_h2 * x2
wx3h2_new = wx3h2 + lr * delta_h2 * x3

b1_new = b1 + lr * delta_h1
b2_new = b2 + lr * delta_h2

print("\nUpdated input-hidden weights:")
print("x1 -> h1:", round(wx1h1_new, 8))
print("x2 -> h1:", round(wx2h1_new, 8))
print("x3 -> h1:", round(wx3h1_new, 8))
print("x1 -> h2:", round(wx1h2_new, 8))
print("x2 -> h2:", round(wx2h2_new, 8))
print("x3 -> h2:", round(wx3h2_new, 8))

print("\nUpdated hidden biases:")
print("b1:", round(b1_new, 8))
print("b2:", round(b2_new, 8))

print("\nUpdated output weights:")
print("h1 -> output:", round(wh1o_new, 8))
print("h2 -> output:", round(wh2o_new, 8))
print("bout:", round(bout_new, 8))