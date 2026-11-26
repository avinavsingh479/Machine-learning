import math

n = int(input("Enter number of features: "))
m = int(input("Enter number of data points: "))
eps = float(input("Enter Epsilon: "))
min_pts = int(input("Enter MinPts: "))

X = []
for i in range(m):
    print(f"\nData point {i+1}:")
    sample = []
    for j in range(n):
        sample.append(float(input(f"Feature {j+1}: ")))
    X.append(sample)


def dist(a, b):
    return math.sqrt(sum((X[a][j] - X[b][j]) ** 2 for j in range(n)))


def region_query(p):
    # indices of all points within eps of p (including p itself)
    return [i for i in range(m) if dist(p, i) <= eps]


visited = set()
clustered = set()          # points already assigned to a cluster
cluster_groups = []
noise = []

for p in range(m):
    if p in visited:
        continue
    visited.add(p)
    neighbors = region_query(p)

    if len(neighbors) < min_pts:
        noise.append(p)            # may become a border point later
    else:
        cluster = []
        queue = neighbors[:]
        while queue:
            q = queue.pop(0)
            if q not in visited:
                visited.add(q)
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_pts:
                    queue.extend(x for x in q_neighbors if x not in queue)
            if q not in clustered:
                cluster.append(q)
                clustered.add(q)
        cluster_groups.append(cluster)

# points first marked as noise but later reached by a cluster are border points
noise = [p for p in noise if p not in clustered]

print("\n========== DBSCAN RESULT ==========")
for i, cluster in enumerate(cluster_groups):
    print(f"Cluster {i+1}: {[X[p] for p in cluster]}")
print("Noise Points:", [X[p] for p in noise])