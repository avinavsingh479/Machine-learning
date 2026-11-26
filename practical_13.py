import math

n = int(input("Enter number of features: "))
m = int(input("Enter number of data points: "))
k = int(input("Enter desired number of clusters: "))

X = []
for i in range(m):
    print(f"\nData point {i+1}:")
    sample = []
    for j in range(n):
        sample.append(float(input(f"Feature {j+1}: ")))
    X.append(sample)

cluster_groups = [[i] for i in range(m)]


def dist(a, b):
    return math.sqrt(sum((X[a][j] - X[b][j]) ** 2 for j in range(n)))


iteration = 1

while len(cluster_groups) > k:
    min_dist = float('inf')
    merge_i, merge_j = 0, 1

    for i in range(len(cluster_groups)):
        for j in range(i + 1, len(cluster_groups)):
            d = min(dist(a, b) for a in cluster_groups[i]
                    for b in cluster_groups[j])
            if d < min_dist:
                min_dist = d
                merge_i, merge_j = i, j

    cluster_groups[merge_i].extend(cluster_groups[merge_j])
    cluster_groups.pop(merge_j)

    print(f"\n========== ITERATION {iteration} ==========")
    print("Merged clusters at single-link distance:", round(min_dist, 8))
    for i, cluster in enumerate(cluster_groups):
        print(f"Cluster {i+1}: {[X[p] for p in cluster]}")
    iteration += 1

print("\n========== FINAL CLUSTERS ==========")
for i, cluster in enumerate(cluster_groups):
    print(f"Cluster {i+1}: {[X[p] for p in cluster]}")