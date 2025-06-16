# === Standard Libraries ===
import locale
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# === Scikit-learn ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight


# === CatBoost ===
from catboost import CatBoostClassifier

# === Hyperparameter Optimization ===
import optuna

# === Local Modules ===
from cm_class import CMThreshold

 
#Select features from X given an array selected_features
def select_features(X, selected_features):
    return np.hstack((X[:,0].reshape(-1,1), X[:,selected_features+1]))

# Custom AUC-PR scoring function
def auc_pr_scorer(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)


def estimate_auc_pr_with_cv(X_train, y_train):
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    auc_pr_scores = []

    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Train a RandomForest model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_tr, y_tr)

        # Predict probabilities on the validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate AUC-PR
        auc_pr = average_precision_score(y_val, y_pred_proba)
        auc_pr_scores.append(auc_pr)

    return np.mean(auc_pr_scores)

#Perform backward feature selection
def backward_feature_selection(X_train, y_train, n_min_features, load_features = False):
    X_train = X_train[:,1:]
    # Initialize the set of selected features with all features
    selected_features = list(range(X_train.shape[1]))
    k = n_min_features

    best_auc_pr = 0
    n_best_auc_pr = -1
    best_selected_features = selected_features
    while len(selected_features) > k:
        print(len(selected_features))
        # Fit a model using the selected features
        X_train_selected = X_train[:, selected_features]
        model = RandomForestClassifier(n_estimators=100)

        model.fit(X_train_selected, y_train)
      
        # Use model performance or feature importance to identify the least significant feature
        # For example, you can use coefficients if using linear models or feature importance for tree-based models
        feature_importance = model.feature_importances_

        n = 1
        # Identify the least important feature
        if len(selected_features) > 1000:
            n = 100
        elif len(selected_features) > 200:
            n = 10
        if(len(selected_features) > 3300):
            n = 1
        smallest_indices = np.argsort(feature_importance)[:n]
        least_important_feature = smallest_indices
        #least_important_feature = np.argmin(feature_importance)

        # Remove the least important feature from the selected features
        index_remove = least_important_feature
        for index in sorted(index_remove, reverse=True):
            selected_features.remove(selected_features[index])

        j = len(selected_features)
        if(j <= 500 and j >= 100 and j%50== 0 ):
            auc_pr = estimate_auc_pr_with_cv(X_train_selected, y_train)
            print("Auc_pr: ", auc_pr)
            if(auc_pr > best_auc_pr):
                best_auc_pr = auc_pr
                n_best_auc_pr = j
                best_selected_features = selected_features.copy()
                
    if(n_best_auc_pr > 0):
        print("Best auc pr: ", best_auc_pr, "for feature_n: ", n_best_auc_pr, " ", len(best_selected_features))
    return np.array(selected_features), np.array(best_selected_features)

#Perform training using a random forest classifier
def perform_training(X_train, y_train, use_class_weight = False):
    clf = RandomForestClassifier(n_estimators=1000)
    if use_class_weight:
         clf = RandomForestClassifier(n_estimators=1000, class_weight = "balanced")
    clf.fit(X_train[:,1:], y_train)
    return clf

#Alternative example how to implement training with other ml models (here: catboost)
def perform_training_catboost(X_train, y_train, use_class_weight = False):
    clf = CatBoostClassifier()
    clf.fit(X_train[:,1:], y_train)
    return clf

#Perform training + hyperparameter tuning using a random forest
def perform_training_rf_cv(X_train, y_train, use_class_weight=False):
    # Remove the first column if needed
    X_train = X_train[:, 1:]

    # Split into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

    # Define number of trials for Optuna
    n_trials = 30

    # Define the objective function for hyperparameter tuning
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step = 100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced" if use_class_weight else None
        }

        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        preds_proba = model.predict_proba(X_valid)[:,1]
        return average_precision_score(y_valid, preds_proba)  # Optimize AUCPR

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Train the best model with optimal hyperparameters
    best_params = study.best_params
    best_model = RandomForestClassifier(**best_params, n_jobs=-1)
    best_model.fit(X_train, y_train)
    print("Best params: ", best_params)
    return best_model

#Calculate performance metrics
def estimate_stats(y_actual, y_pred, y_pred_prob, print_stats=False, show_plots=False):

    accuracy = accuracy_score(y_actual, y_pred)
    classification_rep = classification_report(y_actual, y_pred)
    conf_matrix = confusion_matrix(y_actual, y_pred)

    prob_same = len(np.unique(y_actual)) == 1
    if not prob_same:
        roc = roc_auc_score(y_actual, y_pred_prob)
        auc_pr = average_precision_score(y_actual, y_pred_prob)
        print("ROC:", roc)
        print("AUC-PR:", auc_pr)
    else:
        roc = 0
        auc_pr = 0

    if print_stats:
        print(f'Accuracy: {accuracy:.2f}')
        print('Classification Report:\n', classification_rep)
        print('Confusion Matrix:\n', conf_matrix)

    # Handle special case for single-class confusion matrix
    if conf_matrix.shape == (1, 1):
        w = conf_matrix[0, 0]
        if np.all(y_actual == 0) and np.all(y_pred == 0):
            conf_matrix = np.array([[w, 0], [0, 0]])
        elif np.all(y_actual == 1) and np.all(y_pred == 1):
            conf_matrix = np.array([[0, 0], [0, w]])
        else:
            raise ValueError("Unexpected single-class confusion matrix state")

    if not prob_same and show_plots:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_actual, y_pred_prob)
        #roc_auc = auc(fpr, tpr)

        plt.figure(dpi=300, figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve\n(AUC = {roc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Dummy\nclassifier (AUC = 0.5)")
        plt.xlabel('False positive rate', fontsize=15)
        plt.ylabel('True positive rate\n(sensitivity)', fontsize=15)
        plt.title('ROC curve', fontsize=18)
        plt.legend(loc="lower left", bbox_to_anchor=(1, 0.5))
        plt.gca().set_aspect(0.7)
        plt.subplots_adjust(bottom=0.2)
        plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_actual, y_pred_prob)
        auc_pr_baseline = y_actual.sum() / len(y_actual)

        plt.figure(dpi=300, figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve\n(AUC = {auc_pr:.2f})')
        plt.axhline(y=auc_pr_baseline, color='navy', linestyle='--', label=f'Dummy classifier (AUC = {auc_pr_baseline:.2f})')
        plt.xlabel('Recall (sensitivity)', fontsize=15)
        plt.ylabel('Precision (PPV)', fontsize=15)
        plt.title('Precision-Recall curve', fontsize=18)
        plt.legend(loc="lower left", bbox_to_anchor=(1, 0.5))
        plt.gca().set_aspect(0.7)
        plt.subplots_adjust(bottom=0.2)
        plt.show()

    return accuracy, classification_rep, conf_matrix, roc, auc_pr


def print_feature_importance(feature_importance, feature_names):

    # Zip feature names and their importances together
    feature_importance_dict = dict(zip(feature_names, feature_importance))

    # Sort the dictionary by importance values in descending order
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    count = 0 
    # Print the features with the highest importance
    for feature, importance in sorted_feature_importance:
        count = count + 1
        print(f"{count}: Feature: {feature}, Importance: {importance}")
        if count > 15:
            break

def print_thresholds(y_pred_test_prob, y_test, thresholds):
    cm_grid = []
    for threshold in thresholds:
        # Convert predicted probabilities to binary predictions based on the threshold
        y_pred_binary = (y_pred_test_prob >= threshold).astype(int)
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred_binary)
        if len(cm_grid) == 0:
            cm_grid = np.array([cm])
        else:
            cm_grid = np.vstack((cm_grid, [cm]))
        print("Confusion matrix for threshold {:.2f}:".format(threshold))
        print(cm)
        print()
    cm_thrshld = CMThreshold(thresholds, cm_grid)
    return cm_thrshld

def calc_confidence_intv(data, confidence_level = 0.95):
    # Step 1: Calculate the mean
    mean = np.mean(data)

    # Step 2: Calculate the standard error of the mean (SEM)
    sem = stats.sem(data)

    # Step 3: Determine the critical value for 95% confidence interval
    # For a 95% CI, the critical value is approximately 1.96
    #confidence_level = 0.95
    critical_value = stats.t.ppf((1 + confidence_level) / 2., len(data)-1)

    # Step 4: Calculate the margin of error
    margin_of_error = critical_value * sem
    # Step 5: Calculate the confidence interval
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    return ci_lower, ci_upper

