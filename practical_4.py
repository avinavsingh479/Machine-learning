def euclidean_distance(a, b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return dist ** 0.5


n = int(input("Enter number of records: "))
m = int(input("Enter number of features: "))
features = []
for i in range(m):
    features.append(input("Enter feature name: "))
target_field = input("Enter target variable name: ")
records = []

print("\nEnter dataset values:")
for i in range(n):
    sample = []
    print("\nRecord", i + 1)
    for attribute in features:
        sample.append(float(input(attribute + ": ")))
    label = input(target_field + ": ")
    records.append((sample, label))

k = int(input("\nEnter value of K: "))

print("\nEnter values for a new sample:")
new_sample = []
for attribute in features:
    new_sample.append(float(input(attribute + ": ")))

distance_list = []
for sample, label in records:
    d = euclidean_distance(sample, new_sample)
    distance_list.append((d, label))

distance_list.sort(key=lambda x: x[0])
neighbors = distance_list[:k]

votes = {}
print("\nNearest Neighbors:")
for d, label in neighbors:
    print("Distance:", round(d, 4), "Class:", label)
    votes[label] = votes.get(label, 0) + 1

predicted_label = max(votes, key=votes.get)
print("\nVoting Results:")
for label, count in votes.items():
    print(label + ":", count)

print("\nPredicted Class:", predicted_label)