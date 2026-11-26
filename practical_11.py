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

cluster_centres = X[:k]          # first K points are the initial centroids
iteration = 1

while True:
    cluster_groups = [[] for _ in range(k)]

    # Assignment step
    for point in X:
        distance_list = []
        for centroid in cluster_centres:
            d = math.sqrt(sum((point[j] - centroid[j]) ** 2
                              for j in range(n)))
            distance_list.append(d)
        cluster = distance_list.index(min(distance_list))
        cluster_groups[cluster].append(point)

    # Update step
    new_centroids = []
    for cluster in cluster_groups:
        if cluster:
            centroid = []
            for j in range(n):
                centroid.append(sum(point[j] for point in cluster)
                                / len(cluster))
            new_centroids.append(centroid)
        else:
            # empty cluster keeps its old centroid
            new_centroids.append(cluster_centres[len(new_centroids)])

    print(f"\n========== ITERATION {iteration} ==========")
    for i in range(k):
        print(f"Cluster {i+1}: {cluster_groups[i]}")
        print(f"Centroid {i+1}: {new_centroids[i]}")

    if new_centroids == cluster_centres:
        break
    cluster_centres = new_centroids
    iteration += 1

print("\n========== FINAL CLUSTERS ==========")
for i in range(k):
    print(f"Cluster {i+1}: {cluster_groups[i]}")
    print(f"Final Centroid: {cluster_centres[i]}")