def log2(x):
    # log base 2 without the math module (series expansion of ln)
    if x == 1:
        return 0
    exponent = 0
    while x >= 2:
        x /= 2
        exponent += 1
    while x < 1:
        x *= 2
        exponent -= 1
    z = (x - 1) / (x + 1)
    term = z
    total = term
    for i in range(3, 100, 2):
        term *= z * z
        total += term / i
    ln_x = 2 * total
    z2 = 1 / 3
    term2 = z2
    total2 = term2
    for i in range(3, 100, 2):
        term2 *= z2 * z2
        total2 += term2 / i
    ln2 = 2 * total2
    return exponent + ln_x / ln2


def entropy(records, target_field):
    label_counts = {}
    for row in records:
        label = row[target_field]
        label_counts[label] = label_counts.get(label, 0) + 1
    ent = 0
    total = len(records)
    for count in label_counts.values():
        p = count / total
        ent -= p * log2(p)
    return ent


def information_gain(records, attribute, target_field):
    parent_entropy = entropy(records, target_field)
    unique_values = set(row[attribute] for row in records)
    weighted_entropy = 0
    for value in unique_values:
        partition = [row for row in records if row[attribute] == value]
        weighted_entropy += (len(partition) / len(records)) * \
            entropy(partition, target_field)
    return parent_entropy - weighted_entropy


def majority_class(records, target_field):
    label_counts = {}
    for row in records:
        label = row[target_field]
        label_counts[label] = label_counts.get(label, 0) + 1
    return max(label_counts, key=label_counts.get)


def build_tree(records, features, target_field):
    class_labels = [row[target_field] for row in records]
    if len(set(class_labels)) == 1:
        return {"leaf": class_labels[0]}
    if not features:
        return {"leaf": majority_class(records, target_field)}
    gains = {}
    for attribute in features:
        gains[attribute] = information_gain(records, attribute, target_field)
    best_feature = max(gains, key=gains.get)
    tree = {
        "attribute": best_feature,
        "gain": gains[best_feature],
        "gains": gains,
        "default": majority_class(records, target_field),
        "branches": {}
    }
    unique_values = sorted(set(row[best_feature] for row in records))
    remaining_features = [f for f in features if f != best_feature]
    for value in unique_values:
        partition = [row for row in records if row[best_feature] == value]
        tree["branches"][value] = build_tree(
            partition, remaining_features, target_field)
    return tree


def print_tree(tree, prefix="", root=True, last=True):
    connector = "" if root else ("└── " if last else "├── ")
    if "leaf" in tree:
        print(prefix + connector + "Predict: " + str(tree["leaf"]))
        return
    print(prefix + connector + tree["attribute"] +
          " [IG: " + str(round(tree["gain"], 4)) + "]")
    branch_prefix = prefix if root else prefix + ("    " if last else "│   ")
    branches = list(tree["branches"].items())
    for i, (value, child) in enumerate(branches):
        is_last = i == len(branches) - 1
        branch_connector = "└── " if is_last else "├── "
        print(branch_prefix + branch_connector + str(value))
        child_prefix = branch_prefix + ("    " if is_last else "│   ")
        print_tree(child, child_prefix, False, True)


def display_gains(tree, level=0):
    if "leaf" in tree:
        return
    print("\n" + "  " * level + "Node: " + tree["attribute"])
    for attribute, gain in tree["gains"].items():
        print("  " * level + attribute + ": " + str(round(gain, 4)))
    for child in tree["branches"].values():
        display_gains(child, level + 1)


def predict(tree, sample):
    if "leaf" in tree:
        return tree["leaf"]
    value = sample[tree["attribute"]]
    if value in tree["branches"]:
        return predict(tree["branches"][value], sample)
    return tree["default"]


n = int(input("Enter number of records: "))
m = int(input("Enter number of features: "))
features = []
for i in range(m):
    features.append(input("Enter feature name: ").strip())
target_field = input("Enter target variable name: ").strip()
records = []

print("\nEnter dataset values:")
for i in range(n):
    row = {}
    print("\nRecord", i + 1)
    for attribute in features:
        row[attribute] = input(attribute + ": ").strip()
    row[target_field] = input(target_field + ": ").strip()
    records.append(row)

tree = build_tree(records, features, target_field)
print("\nGenerated ID3 Decision Tree:\n")
print_tree(tree)

print("\nInformation Gain Calculations:")
display_gains(tree)

print("\nEnter values for a new sample:")
sample = {}
for attribute in features:
    sample[attribute] = input(attribute + ": ").strip()
result = predict(tree, sample)
print("\nPredicted Class:", result)