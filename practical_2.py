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




seed = 12345


def random_number():
    # simple linear congruential generator (no random module)
    global seed
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed


def shuffle(items):
    items = items[:]
    for i in range(len(items) - 1, 0, -1):
        j = random_number() % (i + 1)
        items[i], items[j] = items[j], items[i]
    return items


def entropy(rows, target_field):
    label_counts = {}
    for row in rows:
        label = row[target_field]
        label_counts[label] = label_counts.get(label, 0) + 1
    result = 0
    for count in label_counts.values():
        p = count / len(rows)
        result -= p * log2(p)
    return result


def information_gain(rows, attribute, target_field):
    parent_entropy = entropy(rows, target_field)
    groups = {}
    for row in rows:
        groups.setdefault(row[attribute], []).append(row)
    weighted_entropy = 0
    for group in groups.values():
        weighted_entropy += (len(group) / len(rows)) * \
            entropy(group, target_field)
    return parent_entropy - weighted_entropy


def majority_class(rows, target_field):
    label_counts = {}
    for row in rows:
        label = row[target_field]
        label_counts[label] = label_counts.get(label, 0) + 1
    return max(label_counts, key=label_counts.get)


def sqrt_integer(n):
    result = 1
    while result * result <= n:
        result += 1
    return result - 1


def build_tree(rows, features, target_field, max_features):
    class_labels = [row[target_field] for row in rows]
    if len(set(class_labels)) == 1:
        return {"leaf": class_labels[0]}
    if not features:
        return {"leaf": majority_class(rows, target_field)}
    selected = shuffle(features)[:max_features]
    gains = {}
    for attribute in selected:
        gains[attribute] = information_gain(rows, attribute, target_field)
    best_feature = max(gains, key=gains.get)
    tree = {"attribute": best_feature, "gain": gains[best_feature],
            "gains": gains, "default": majority_class(rows, target_field),
            "branches": {}}
    remaining = [f for f in features if f != best_feature]
    for value in sorted(set(row[best_feature] for row in rows)):
        partition = [row for row in rows if row[best_feature] == value]
        tree["branches"][value] = build_tree(
            partition, remaining, target_field, max_features)
    return tree


def print_tree(tree, prefix=""):
    if "leaf" in tree:
        print(prefix + "Predict: " + str(tree["leaf"]))
        return
    print(prefix + tree["attribute"] + " [IG: " +
          str(round(tree["gain"], 4)) + "]")
    branches = list(tree["branches"].items())
    for i, (value, child) in enumerate(branches):
        last = i == len(branches) - 1
        print(prefix + ("└── " if last else "├── ") + str(value))
        print_tree(child, prefix + ("    " if last else "│   "))


def display_gains(tree):
    if "leaf" in tree:
        return
    print("\nNode:", tree["attribute"])
    for attribute, gain in tree["gains"].items():
        print(attribute + ":", round(gain, 4))
    for child in tree["branches"].values():
        display_gains(child)


def bootstrap_sample(rows):
    sample = []
    for i in range(len(rows)):
        sample.append(rows[random_number() % len(rows)])
    return sample


def predict(tree, sample):
    if "leaf" in tree:
        return tree["leaf"]
    value = sample[tree["attribute"]]
    if value in tree["branches"]:
        return predict(tree["branches"][value], sample)
    return tree["default"]


def forest_predict(forest, sample):
    votes = {}
    for tree in forest:
        result = predict(tree, sample)
        votes[result] = votes.get(result, 0) + 1
    return max(votes, key=votes.get)


print("Random Forest Classification")
n = int(input("Enter number of records: "))
f = int(input("Enter number of features: "))
feature_names = []
for i in range(f):
    feature_names.append(input("Enter feature name: ").strip())
target_name = input("Enter target variable name: ").strip()
rows = []

print("\nEnter dataset values:")
for i in range(n):
    row = {}
    print("\nRecord", i + 1)
    for attribute in feature_names:
        row[attribute] = input(attribute + ": ").strip()
    row[target_name] = input(target_name + ": ").strip()
    rows.append(row)

num_trees = int(input("\nEnter number of trees: "))
max_features = sqrt_integer(f)
if max_features < 1:
    max_features = 1

forest = []
for i in range(num_trees):
    sample = bootstrap_sample(rows)
    tree = build_tree(sample, feature_names, target_name, max_features)
    forest.append(tree)

print("\nGenerated Random Forest:")
for i, tree in enumerate(forest):
    print("\nTree", i + 1)
    print_tree(tree)

print("\nInformation Gain Calculations:")
for i, tree in enumerate(forest):
    print("\nTree", i + 1)
    display_gains(tree)

print("\nEnter values for a new sample:")
sample = {}
for attribute in feature_names:
    sample[attribute] = input(attribute + ": ").strip()

print("\nIndividual Tree Predictions:")
for i, tree in enumerate(forest):
    print("Tree", i + 1, ":", predict(tree, sample))

predicted_label = forest_predict(forest, sample)
print("\nFinal Random Forest Prediction:", predicted_label)