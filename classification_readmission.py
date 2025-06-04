#===Imports===
#import sys
import pickle
import numpy as np
import pandas as pd
#import seaborn as sns
#import torch
import shap
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


from helper_func import (
    estimate_stats, print_feature_importance, print_thresholds,
    backward_feature_selection, select_features, calc_confidence_intv,
    perform_training_catboost, perform_training
)

#===Config===
calculate_feature_importance = True
save_confusion_matrices = False
use_smote = False
num_features = 100
iterations = 3
np.random.seed(42)

#===LoadDataSet===
df = pd.read_csv('../metadata_at_readmission.csv')
target_column = "Komplikation_Wiederaufnahme"
X = df.drop(columns=[target_column])
y = df[target_column].values

print("Feature shape:", X.shape)
print("Target positives:", y.sum())

#===InitMetrics===
grid_aucroc = []
grid_aucpr = []
grid_baseline = []
cm_grid = []

#===MainLoop===
for i in range(iterations):
    print(f"\n=== Iteration {i+1}/{iterations} ===")

    #Split in training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=50/len(y), random_state=42*i
    )
    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

    #Apply scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Perform feature selection
    print("Performing backward selection: ")
    selected_features, _ = backward_feature_selection(X_train, y_train, num_features)

    #Smote
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    #Apply feature selection
    X_train = select_features(X_train, selected_features)
    X_test = select_features(X_test, selected_features)

    #Train model
    model = perform_training(X_train, y_train)
    #model = perform_training_catboost(X_train, y_train)

    #===TrainEval===
    y_pred_train = model.predict(X_train[:, 1:])
    y_proba_train = model.predict_proba(X_train[:, 1:])[:, 1]
    _, _, cm_train, auc_roc_train, auc_pr_train = estimate_stats(
        y_train, y_pred_train, y_proba_train, print_stats=True, show_plots=False
    )
    print("\nTrain Metrics")
    print("Confusion Matrix:\n", cm_train)
    print("AUC-ROC:", auc_roc_train)
    print("AUC-PR:", auc_pr_train)

    #===TestEval===
    y_pred = model.predict(X_test[:, 1:])
    y_proba = model.predict_proba(X_test[:, 1:])[:, 1]
    _, _, cm, auc_roc, auc_pr = estimate_stats(
        y_test, y_pred, y_proba, print_stats=True, show_plots=False
    )
    print("\nTest Metrics")
    print("Confusion Matrix:\n", cm)
    print("AUC-ROC:", auc_roc)
    print("AUC-PR:", auc_pr)

    #===FeatureImportance===
    if calculate_feature_importance:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train, approximate=True)
        feature_importances = np.abs(shap_values).mean(axis=0)[:, 0]
        feature_names = X.columns[selected_features + 1]
        model_importances = model.feature_importances_

        feature_data = list(zip(feature_importances, model_importances, feature_names))
        sorted_features = sorted(feature_data, key=lambda x: x[0], reverse=True)

        for shap_val, importance, name in sorted_features:
            print(f"{shap_val:.4f} === {importance:.4f} === {name}")
        print_feature_importance(feature_importances, feature_names)

    #===RecordMetrics===
    grid_aucroc.append(auc_roc)
    grid_aucpr.append(auc_pr)
    grid_baseline.append(np.mean(y_test))

    print("\nConfusion matrices for diferent thresholds:")
    thresholds_cm = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75]
    cm_grid.append(print_thresholds(y_proba, y_test, thresholds_cm))
    print("Iteration finished")


# === Summary ===
print("\nSummary:")
sum_cm = 0
n = len(cm_grid)
for j in range(n):
        if j == 0:
            sum_cm = cm_grid[0].cm
        else:
            sum_cm = sum_cm + cm_grid[j].cm
sum_cm = sum_cm/n

confusion_matrices = {}

print("\nSummary:")
print("Averaged confusion matrices:\n")
for j in range(len(sum_cm)):
    print("\nThreshold: ", thresholds_cm[j], "\n\n", sum_cm[j])
    confusion_matrices[thresholds_cm[j]] = sum_cm[j]

#Calculate stats
aucroc_lower, aucroc_upper = calc_confidence_intv(grid_aucroc)
aucpr_lower, aucpr_upper = calc_confidence_intv(grid_aucpr)
baseline_lower, baseline_upper = calc_confidence_intv(grid_baseline)

if len(grid_aucroc) > 1:
    print("\nAUC-ROC:", np.mean(grid_aucroc), f"CI: [{aucroc_lower:.4f}, {aucroc_upper:.4f}]")
    print("AUC-PR:", np.mean(grid_aucpr), f"CI: [{aucpr_lower:.4f}, {aucpr_upper:.4f}]")
    print("Baseline:", np.mean(grid_baseline), f"CI: [{baseline_lower:.4f}, {baseline_upper:.4f}]")
else:
    print("\nAUC-ROC:", np.mean(grid_aucroc))
    print("AUC-PR:", np.mean(grid_aucpr))
    print("Baseline:", np.mean(grid_baseline))

if save_confusion_matrices:
    with open('confusion_matrices.pkl', 'wb') as f:
        pickle.dump(confusion_matrices, f)
