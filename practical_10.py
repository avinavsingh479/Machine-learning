import math

n = int(input("Enter number of features: "))
m = int(input("Enter number of data points: "))

X = []
Y = []

for i in range(m):
    print(f"\nData point {i+1}:")
    sample = []
    for j in range(n):
        sample.append(float(input(f"Feature {j+1}: ")))
    X.append(sample)
    Y.append(input("Class label: "))

print("\nEnter new data point:")
new_point = []
for i in range(n):
    new_point.append(float(input(f"Feature {i+1}: ")))

k = int(input("Enter K: "))

distance_list = []
for i in range(m):
    d = 0
    for j in range(n):
        d += (new_point[j] - X[i][j]) ** 2
    d = math.sqrt(d)
    distance_list.append((d, X[i], Y[i]))

distance_list.sort(key=lambda x: x[0])

print("\n========== DISTANCE CALCULATION ==========")
print(f"{'Point':<25}{'Distance':<15}{'Class':<10}")
print("-" * 50)
for d, point, label in distance_list:
    print(f"{str(point):<25}{d:<15.8f}{label:<10}")

neighbors = distance_list[:k]

print("\n========== NEAREST NEIGHBORS ==========")
print(f"{'Point':<25}{'Distance':<15}{'Class':<10}")
print("-" * 50)
for d, point, label in neighbors:
    print(f"{str(point):<25}{d:<15.8f}{label:<10}")

votes = {}
for d, point, label in neighbors:
    votes[label] = votes.get(label, 0) + 1

print("\n========== VOTING ==========")
for label, count in votes.items():
    print(f"Class {label}: {count} vote(s)")

predicted_label = max(votes, key=votes.get)

print("\n========== RESULT ==========")
print("Predicted Class:", predicted_label)