# === Imports ===
import random
import pickle
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt

#from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE


# === Local Modules ===
from helper_func import (
    perform_training, perform_training_rf_cv,
    perform_training_catboost, estimate_stats,
    backward_feature_selection, select_features,
    print_thresholds, print_feature_importance, calc_confidence_intv
)


# === Settings ===

# Number of iterations, for random subsampling, bootstrapping and testing on LLM labels this should be > 1
iterations = 1

# Size of test set
num_test_samples = 500

# Minimum number of features
min_num_features = 100

# Test on validation set (train using LLM-labeled training set)
test_on_validation_set = True

# Perform random subsampling on the validation set only
test_on_valid_random_subsampling = False

# Perform bootstrapping on the validation set only
test_on_valid_bootstrapping = False


# Select number of features that maximize AUC-PR within the training set
select_n_best_features = True

# Save confusion matrices
save_confusion_matrices = False

# Use Gini importance instead of SHAP
use_gini_importance = False

#Use random seed
#np.random.seed(43)

#Use SMOTE
use_smote = False

if test_on_valid_bootstrapping and test_on_valid_random_subsampling:
    raise Exception("Both cross validation and bootstrapping are activated. Choose only one.")

# ==================================



# === Load Data ===
result_df = pd.read_csv('../metadata_at_discharge.csv')

# Choose complication type
#comp = "kompart_Blutung_combined"
comp = "kompart_Perforation_combined"

# Validation: Endoscopy reports
validation_df_endo = pd.read_excel("../gt_endoscopy_reports.xlsx")
validation_df_endo["komplikationsart"] = validation_df_endo['komplikationsart'].fillna('')
validation_df_endo["kompart_Perforation_endo_valid"] = validation_df_endo['komplikationsart'].str.contains('Perforation', case=False).astype(int)
validation_df_endo["kompart_Blutung_endo_valid"] = validation_df_endo['komplikationsart'].str.contains('Blutung', case=False).astype(int)
validation_df_endo = validation_df_endo[["falnr", "kompart_Blutung_endo_valid", "kompart_Perforation_endo_valid"]].drop_duplicates()

# Validation: Discharge letters
validation_df_arztbrief = pd.read_excel("../gt_discharge_letters.xlsx")
validation_df_arztbrief["kompart_Perforation_arztbrief_valid"] = (validation_df_arztbrief["kompart_Perforation"] == 1).astype(int)
validation_df_arztbrief["kompart_Blutung_arztbrief_valid"] = (validation_df_arztbrief["kompart_Blutung"] == 1).astype(int)
validation_df_arztbrief = validation_df_arztbrief[["encounter_id", "kompart_Perforation_arztbrief_valid", "kompart_Blutung_arztbrief_valid"]]

# Merge validation sources
validation_df = pd.merge(validation_df_endo, validation_df_arztbrief, left_on="falnr", right_on="encounter_id", how="left").fillna(0)
validation_df["kompart_Blutung_combined_valid"] = validation_df["kompart_Blutung_endo_valid"].astype(int) | validation_df["kompart_Blutung_arztbrief_valid"].astype(int)
validation_df["kompart_Perforation_combined_valid"] = validation_df["kompart_Perforation_endo_valid"].astype(int) | validation_df["kompart_Perforation_arztbrief_valid"].astype(int)
validation_df["kompart_validation"] = validation_df[f"{comp}_valid"]

# Drop unused columns
val_cols_to_drop = [
    "kompart_Blutung_arztbrief_valid", "kompart_Perforation_endo_valid",
    "kompart_Perforation_arztbrief_valid", "kompart_Blutung_combined_valid",
    "kompart_Perforation_combined_valid"
]
validation_df = validation_df.drop(columns=val_cols_to_drop)

# Merge validation with main dataset
if test_on_validation_set:
    result_df = pd.merge(result_df, validation_df[["falnr", "kompart_validation"]], on="falnr", how='left')
    #result_df["kompart_validation"].fillna(-1, inplace=True)
    result_df["kompart_validation"] = result_df["kompart_validation"].fillna(-1)
    validation_df = result_df[result_df["kompart_validation"].isin([0, 1])]
    result_df = result_df[result_df["kompart_validation"] == -1]

result_df.drop_duplicates(inplace=True)
result_df = result_df.reset_index(drop=True)

# === Prepare Labels ===
y = result_df[comp].values.astype(int)

if test_on_validation_set:
    y_train_val = y
    y_test_val = validation_df["kompart_validation"].values.astype(int)
    y_val_llm = validation_df[comp].values.astype(int)

# Drop complication columns
columns_to_drop = [
    "kompart_Blutung_endo", "kompart_Perforation_endo",
    "kompart_Blutung_arztbrief", "kompart_Perforation_arztbrief",
    "kompart_Blutung_combined", "kompart_Perforation_combined"
]

result_df.drop(columns=columns_to_drop, inplace=True)

if test_on_validation_set:
    result_df.drop(columns="kompart_validation", inplace=True)
    validation_df.drop(columns=["kompart_validation"] + columns_to_drop, inplace=True)

result_df.fillna(-1, inplace=True)

X = result_df.values

# === Main Training Loop ===
grid_aucpr, grid_roc, grid_baseline = [], [], []
cm_grid = []
test_size_ = num_test_samples / result_df.shape[0]



for i in range(iterations):  # One run for now
    print(f"Run# {i}")

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, range(len(X)), test_size=test_size_)

    if test_on_validation_set and not test_on_valid_random_subsampling and not test_on_valid_bootstrapping:
        X_train = result_df.values
        X_test = validation_df.values
        y_train = y_train_val
        y_test = y_test_val

    elif test_on_valid_random_subsampling:
        X_random_subsampling = validation_df.values
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X_random_subsampling, y_test_val, range(len(X_random_subsampling)))
        y_train = y_val_llm[indices_train]

    elif test_on_valid_bootstrapping:
        X_all = validation_df.values
        n_samples = 500
        indices_train = np.random.choice(n_samples, size=n_samples, replace=True)
        indices_test = np.setdiff1d(np.arange(n_samples), indices_train)
        X_train, y_train = X_all[indices_train], y_test_val[indices_train]
        X_test, y_test = X_all[indices_test], y_test_val[indices_test]

    print("Shape train:", X_train.shape, "| test:", X_test.shape)

    # Apply SMOTE only on training data
    if(use_smote):
        smote = SMOTE(sampling_strategy='auto', random_state=42)  # Adjust strategy if needed
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # --- Scale ---
    scaler = StandardScaler()
    X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])
    X_test[:, 1:] = scaler.transform(X_test[:, 1:])

    # --- Feature Selection ---
    print("Performing backward selection:")
    selected_features, best_selected_features = backward_feature_selection(X_train, y_train, min_num_features)
    selected_features = best_selected_features if select_n_best_features else selected_features

    X_train = select_features(X_train, selected_features)
    X_test = select_features(X_test, selected_features)

    # --- Train Model ---
    clf = perform_training(X_train, y_train)
    #clf = perform_training_rf_cv(X_train, y_train)
    #clf = perform_training_catboost(X_train, y_train, False)

    print("Number of selected features:", len(selected_features))
    feature_names = result_df.columns[selected_features + 1]

    # --- Feature Importance ---
    if use_gini_importance and hasattr(clf, "feature_importances_"):
        feature_importances = clf.feature_importances_
    else:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_train, approximate=True)
        feature_importances = np.abs(shap_values).mean(axis=0)[:, 0]

    print_feature_importance(feature_importances, feature_names)

    # --- Predict and Evaluate ---
    y_pred_train = clf.predict(X_train[:, 1:])
    y_pred_train_prob = clf.predict_proba(X_train[:, 1:])[:, 1]

    y_pred_test = clf.predict(X_test[:, 1:])
    y_pred_test_prob = clf.predict_proba(X_test[:, 1:])[:, 1]

    print("\nTraining:")
    estimate_stats(y_train, y_pred_train, y_pred_train_prob, print_stats = True, show_plots = False)
    
    print("\nTesting:")
    _, _, conf_matrix, roc, auc_pr = estimate_stats(y_test, y_pred_test, y_pred_test_prob, print_stats=True, show_plots=False)

    grid_roc.append(roc)
    grid_aucpr.append(auc_pr)
    grid_baseline.append(np.sum(y_test)  / len(y_test))

    thresholds_cm = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75]
    print("\nConfusion matrices for diferent thresholds:")
    cm_grid.append(print_thresholds(y_pred_test_prob, y_test, thresholds_cm))
    print("Iteration finished")


# === Summary ===
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


aucroc_lower, aucroc_upper = calc_confidence_intv(grid_roc)
aucpr_lower, aucpr_upper = calc_confidence_intv(grid_aucpr)
baseline_lower, baseline_upper = calc_confidence_intv(grid_baseline)

if len(grid_roc) > 1:
    print("\nAUC-ROC:", np.mean(grid_roc), f"CI: [{aucroc_lower:.4f}, {aucroc_upper:.4f}]")
    print("AUC-PR:", np.mean(grid_aucpr), f"CI: [{aucpr_lower:.4f}, {aucpr_upper:.4f}]")
    print("Baseline:", np.mean(grid_baseline), f"CI: [{baseline_lower:.4f}, {baseline_upper:.4f}]")
else:
    print("\nAUC-ROC:", np.mean(grid_roc))
    print("AUC-PR:", np.mean(grid_aucpr))
    print("Baseline:", np.mean(grid_baseline))

if save_confusion_matrices:
    with open('confusion_matrices.pkl', 'wb') as f:
        pickle.dump(confusion_matrices, f)
