import math

n = int(input("Enter number of features: "))
m = int(input("Enter number of data points: "))
k = int(input("Enter number of clusters (K): "))

X = []
for i in range(m):
    print(f"\nData point {i+1}:")
    sample = []
    for j in range(n):
        sample.append(float(input(f"Feature {j+1}: ")))
    X.append(sample)

if k > m:
    raise SystemExit("K cannot be greater than the number of data points.")

medoids = X[:k]                  # first K points are the initial medoids
iteration = 1

while True:
    cluster_groups = [[] for _ in range(k)]

    # Assignment step
    for point in X:
        distance_list = []
        for medoid in medoids:
            d = math.sqrt(sum((point[j] - medoid[j]) ** 2
                              for j in range(n)))
            distance_list.append(d)
        cluster = distance_list.index(min(distance_list))
        cluster_groups[cluster].append(point)

    # Update step: the new medoid is the member with the least total distance
    new_medoids = []
    for cluster in cluster_groups:
        if cluster:
            costs = []
            for candidate in cluster:
                cost = sum(math.sqrt(sum((candidate[j] - point[j]) ** 2
                                         for j in range(n)))
                           for point in cluster)
                costs.append(cost)
            new_medoids.append(cluster[costs.index(min(costs))])
        else:
            new_medoids.append(medoids[len(new_medoids)])

    print(f"\n========== ITERATION {iteration} ==========")
    for i in range(k):
        print(f"Cluster {i+1}: {cluster_groups[i]}")
        print(f"Medoid {i+1}: {new_medoids[i]}")

    if new_medoids == medoids:
        break
    medoids = new_medoids
    iteration += 1

print("\n========== FINAL CLUSTERS ==========")
for i in range(k):
    print(f"Cluster {i+1}: {cluster_groups[i]}")
    print(f"Final Medoid: {medoids[i]}")