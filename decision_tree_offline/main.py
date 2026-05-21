import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from decision_tree import build_decision_tree, predict, predict_proba
from randomForest import random_forest_train, random_forest_predict, random_forest_predict_proba
from extraTree import extra_trees_train, extra_trees_predict, extra_trees_predict_proba

def load_dataset(name):
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    else:
        raise ValueError("Unsupported dataset name.")
    return data.data, data.target, data.feature_names

def evaluate_decision_tree(X_test, y_test, root):
    y_pred = predict(X_test, root)
    y_prob = predict_proba(X_test, root, len(np.unique(y_test)))
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auroc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    return accuracy, f1, auroc

def evaluate_rf_ensemble(X_test, rf, y, y_test):
    y_pred = random_forest_predict(X_test, rf)
    y_prob = random_forest_predict_proba(X_test, rf, len(np.unique(y)))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    return acc, f1, auc

def evaluate_et_ensemble(X_test, et, y, y_test):
    y_pred = extra_trees_predict(X_test, et)
    y_prob = extra_trees_predict_proba(X_test, et, len(np.unique(y)))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    return acc, f1, auc

def evaluate_sklearn_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auroc = roc_auc_score(y_test, y_prob, multi_class="ovr")

    return accuracy, f1, auroc

def print_sklearn_results(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    accuracy, f1, auroc = evaluate_sklearn_model(dt, X_test, y_test)
    print(f"Sk-learn Decision Tree:    {accuracy:.4f}      {f1:.4f}    {auroc:.4f}")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy, f1, auroc = evaluate_sklearn_model(rf, X_test, y_test)
    print(f"Sk-learn Random Forest:    {accuracy:.4f}      {f1:.4f}    {auroc:.4f}")

    et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et.fit(X_train, y_train)
    accuracy, f1, auroc = evaluate_sklearn_model(et, X_test, y_test)
    print(f"Sk-learn Extra Tree   :    {accuracy:.4f}      {f1:.4f}    {auroc:.4f}")

rng = np.random.RandomState(7)
test_data_size = 0.3
iris_max_depth = 2
wine_max_depth = 6
num_trees = 5
min_sample_per_split = 10

X, y, feature_names = load_dataset("iris")
n_features = int(np.sqrt(X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data_size, random_state=rng)
print("Iris dataset:")
print("                         Accuracy    F1-Score     AUROC")

# ---------- Decision Tree ----------
root = build_decision_tree(X_train, y_train, depth=0, max_tree_depth = iris_max_depth, is_random_split=False, rng=rng)
acc, f1, auc = evaluate_decision_tree(X_test, y_test, root)
print(f"Custom Decision Tree  :    {acc:.4f}      {f1:.4f}    {auc:.4f}")
# ---------- Random Forest ----------
rf = random_forest_train(X_train, y_train, n_trees= num_trees, max_tree_depth= iris_max_depth, min_samples_split= min_sample_per_split, n_features=n_features)
rf_acc, rf_f1, rf_auc = evaluate_rf_ensemble(X_test, rf, y, y_test)
print(f"Custom Random Forest  :    {rf_acc:.4f}      {rf_f1:.4f}    {rf_auc:.4f}")
# ---------- Extra Trees ----------
et = extra_trees_train(X_train, y_train, n_trees= num_trees, max_tree_depth= iris_max_depth, n_features=n_features)
et_acc, et_f1, et_auc = evaluate_et_ensemble(X_test, et, y, y_test)
print(f"Custom Extra Trees    :    {et_acc:.4f}      {et_f1:.4f}    {et_auc:.4f}")
print_sklearn_results(X_train, X_test, y_train, y_test)



X, y, feature_names = load_dataset("wine")
n_features = int(np.sqrt(X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data_size, random_state=rng)
print("\nWine dataset:")
print("                         Accuracy    F1-Score     AUROC")

# ---------- Decision Tree ----------
root = build_decision_tree(X_train, y_train, depth=0, max_tree_depth= wine_max_depth, is_random_split=False, rng=rng)
acc, f1, auc = evaluate_decision_tree(X_test, y_test, root)
print(f"Custom Decision Tree  :    {acc:.4f}      {f1:.4f}    {auc:.4f}")

# ---------- Random Forest ----------
rf = random_forest_train(X_train, y_train, n_trees= num_trees, max_tree_depth= wine_max_depth, min_samples_split=min_sample_per_split, n_features=n_features)
rf_acc, rf_f1, rf_auc = evaluate_rf_ensemble(X_test, rf, y, y_test)
print(f"Custom Random Forest  :    {rf_acc:.4f}      {rf_f1:.4f}    {rf_auc:.4f}")

# ---------- Extra Trees ----------
et = extra_trees_train(X_train, y_train, n_trees= num_trees, max_tree_depth= wine_max_depth, n_features=n_features)
et_acc, et_f1, et_auc = evaluate_et_ensemble(X_test, et, y, y_test)
print(f"Custom Extra Trees    :    {et_acc:.4f}      {et_f1:.4f}    {et_auc:.4f}")

print_sklearn_results(X_train, X_test, y_train, y_test)
