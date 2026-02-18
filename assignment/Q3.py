import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

current_script_path = Path(__file__).resolve().parent
data_file_path = current_script_path.parent / "data" / "zoo.csv"

# 1. Load the data and perform EDA
zoo_data = pd.read_csv(data_file_path)

print("Data dimensions:", zoo_data.shape)
print("\nHead of the data:")
print(zoo_data.head())

# Class sizes
print("\nClass sizes:")
print(zoo_data["class"].value_counts().sort_index())

# Print the animals in each class (to get a sense of groups)
print("\nAnimals by class:")
for class_label in sorted(zoo_data["class"].unique()):
    animals_in_class = zoo_data.loc[zoo_data["class"] == class_label, "animal"].tolist()
    print(f"Class {class_label} ({len(animals_in_class)} animals): {animals_in_class}")

# Feature variation summary
feature_columns = [column for column in zoo_data.columns if column not in ["animal", "class"]]
print("\nFeature variation summary (mean/std/min/max):")
print(zoo_data[feature_columns].agg(["mean", "std", "min", "max"]).T)


# 2. Split the data 50/50 into training and test sets (stratified)
X_all_features = zoo_data[feature_columns]
y_class_label = zoo_data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X_all_features,
    y_class_label,
    test_size=0.50,
    random_state=42,
    stratify=y_class_label
)

print("\nTrain/test split sizes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("\nClass distribution in TRAIN:")
print(y_train.value_counts().sort_index())
print("\nClass distribution in TEST:")
print(y_test.value_counts().sort_index())


# 3. Build a kNN classifier using all variables and select k using cross-validation
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

candidate_k_values = list(range(1, 16, 2))  

# Dataset is small and some classes are tiny, so 3-fold CV is more stable here
cross_validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

cross_validation_results = []
for k in candidate_k_values:
    knn_pipeline.set_params(knn__n_neighbors=k)
    cv_scores = cross_val_score(
        knn_pipeline,
        X_train,
        y_train,
        cv=cross_validation,
        scoring="accuracy"
    )
    cross_validation_results.append((k, cv_scores.mean(), cv_scores.std()))

cross_validation_results_sorted = sorted(cross_validation_results, key=lambda x: x[1], reverse=True)
best_k = cross_validation_results_sorted[0][0]

print("\nCross-validation accuracy on TRAIN set for candidate k values:")
for k, mean_accuracy, standard_deviation in cross_validation_results_sorted:
    print(f"k={k:>2}: mean CV accuracy={mean_accuracy:.3f} (std={standard_deviation:.3f})")

print(f"\nSelected k (best mean CV accuracy): {best_k}")


# 4. Confusion table for the optimal model on the test set + accuracy
knn_pipeline.set_params(knn__n_neighbors=best_k)
knn_pipeline.fit(X_train, y_train)

predicted_test_classes = knn_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, predicted_test_classes)

print(f"\nTest accuracy (all features, k={best_k}): {test_accuracy:.3f}")

labels = sorted(y_class_label.unique())
confusion_all_features = confusion_matrix(y_test, predicted_test_classes, labels=labels)
confusion_table_all_features = pd.DataFrame(
    confusion_all_features,
    index=[f"Actual {label}" for label in labels],
    columns=[f"Pred {label}" for label in labels]
)

print("\nConfusion table (all features):")
print(confusion_table_all_features)


# 5. kNN classifier using only milk, aquatic, airborne + confusion table + predict_proba
subset_feature_names = ["milk", "aquatic", "airborne"]
X_subset_features = zoo_data[subset_feature_names]

X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(
    X_subset_features,
    y_class_label,
    test_size=0.50,
    random_state=42,
    stratify=y_class_label
)

knn_subset_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])

knn_subset_pipeline.fit(X_train_subset, y_train_subset)

predicted_subset_classes = knn_subset_pipeline.predict(X_test_subset)
subset_test_accuracy = accuracy_score(y_test_subset, predicted_subset_classes)

print(f"\nTest accuracy (milk+aqua+airborne, k={best_k}): {subset_test_accuracy:.3f}")

confusion_subset = confusion_matrix(y_test_subset, predicted_subset_classes, labels=labels)
confusion_table_subset_features = pd.DataFrame(
    confusion_subset,
    index=[f"Actual {label}" for label in labels],
    columns=[f"Pred {label}" for label in labels]
)

print("\nConfusion table (milk, aquatic, airborne only):")
print(confusion_table_subset_features)

# Predict probabilities (to see whether all classes are represented in predict_proba)
predicted_probabilities = knn_subset_pipeline.predict_proba(X_test_subset.values)

print("\nProbability output shape (rows = test animals, columns = classes):", predicted_probabilities.shape)
print("Classes used by the model (columns in predict_proba):", knn_subset_pipeline.named_steps["knn"].classes_)

unique_predicted_labels = sorted(pd.Series(predicted_subset_classes).unique().tolist())
print("Unique predicted class labels (subset model):", unique_predicted_labels)
