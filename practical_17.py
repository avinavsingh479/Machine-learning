import numpy as np

m = int(input("Enter number of data points: "))
n = int(input("Enter number of features: "))
k = int(input("Enter number of principal components: "))

X = []
for i in range(m):
    print(f"\nData point {i+1}:")
    sample = []
    for j in range(n):
        sample.append(float(input(f"Feature {j+1}: ")))
    X.append(sample)

X = np.array(X)

# 1. Centre the data
feature_mean = np.mean(X, axis=0)
X_centered = X - feature_mean

# 2. Covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# 3. Eigen-decomposition (eigh is for symmetric matrices)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 4. Sort components by decreasing eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. Project onto the first K components
selected_vectors = eigenvectors[:, :k]
X_pca = X_centered @ selected_vectors

print("\n========== COVARIANCE MATRIX ==========")
print(cov_matrix)

print("\n========== EIGENVALUES ==========")
print(eigenvalues)

print("\n========== EIGENVECTORS ==========")
print(eigenvectors)

print("\n========== REDUCED DATA ==========")
print(X_pca)