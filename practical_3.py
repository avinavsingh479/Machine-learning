def naive_bayes(records, features, target_field, new_sample):
    classes = set(row[target_field] for row in records)
    total = len(records)
    class_probabilities = {}
    scores = {}
    for cls in classes:
        class_rows = [row for row in records if row[target_field] == cls]
        class_probabilities[cls] = len(class_rows) / total
        score = class_probabilities[cls]
        for attribute in features:
            unique_values = set(row[attribute] for row in records)
            # count rows of this class that match the new sample's value
            count = sum(1 for row in class_rows
                        if row[attribute] == new_sample[attribute])
            # Laplace smoothing
            score *= (count + 1) / (len(class_rows) + len(unique_values))
        scores[cls] = score
    return scores, max(scores, key=scores.get)


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

print("\nEnter values for a new sample:")
new_sample = {}
for attribute in features:
    new_sample[attribute] = input(attribute + ": ").strip()

scores, predicted_label = naive_bayes(records, features, target_field,
                                      new_sample)
print("\nClass Scores:")
for cls, score in scores.items():
    print(cls + ":", score)

print("\nPredicted Class:", predicted_label)