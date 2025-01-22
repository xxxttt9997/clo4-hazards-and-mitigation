"""
Complete Model Pipeline

Sections:
    1. KNN Imputation (Optional)
    2. Preliminary Model Selection & Evaluation
    3. Bayesian Optimization (CatBoost Example)
    4. Final CatBoost Model Training & Evaluation
    5. Feature Importances (CatBoost + SHAP)
    6. T-Test on Features
    7. Example Model with Top Features + SHAP Visualization

All code is in English, and comments are included to guide usage.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Sklearn Imports
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
    make_scorer
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# CatBoost
from catboost import CatBoostClassifier, Pool

# Bayesian Optimization
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events

# T-Test
from scipy.stats import ttest_ind

# Saving models/scalers
from joblib import dump
from scipy.ndimage import gaussian_filter1d


# -------------------------------------------------------------------------
# 1. KNN Imputation (Optional)
# -------------------------------------------------------------------------
def knn_imputation(input_csv: str, output_csv: str, n_neighbors: int = 5) -> None:
    """
    Reads a CSV file, performs KNN-based imputation, and saves the result.
    :param input_csv: path to the CSV with missing values
    :param output_csv: path to save the imputed CSV
    :param n_neighbors: number of neighbors to use in KNNImputer
    """
    df_for_knn = pd.read_csv(input_csv)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_filled_knn = pd.DataFrame(
        imputer.fit_transform(df_for_knn),
        columns=df_for_knn.columns
    )
    df_filled_knn.to_csv(output_csv, index=False)
    print(f"KNN Imputation done. Saved to {output_csv}")


# -------------------------------------------------------------------------
# 2. Preliminary Model Selection & Evaluation (Multiple Models)
# -------------------------------------------------------------------------
def model_selection_pipeline(csv_path: str) -> None:
    """
    Loads a dataset, binarizes the target (>=70 = 1), trains multiple models,
    and evaluates performance on a hold-out test set. Also performs 10-fold CV.
    :param csv_path: path to the CSV file containing data
    """
    # Load data
    df = pd.read_csv(csv_path)
    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    y_binary = (df.iloc[:, -1] >= 70).astype(int)

    # Drop columns if not needed
    X = df.iloc[:, :-1].drop(columns=['Year'], errors='ignore')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=99, stratify=y_binary
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Define candidate models
    models = {
        "Random Forest": RandomForestClassifier(
            random_state=99, class_weight='balanced'
        ),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric='logloss',
            random_state=99,
            scale_pos_weight=class_weights[1] / class_weights[0]
        ),
        "CatBoost": CatBoostClassifier(
            verbose=0, random_state=99, auto_class_weights='Balanced'
        ),
        "SVM": SVC(
            probability=True, random_state=99, class_weight='balanced'
        ),
        "MLP": MLPClassifier(random_state=99)
    }

    def print_performance_metrics(y_true, y_pred):
        """Helper function to print performance metrics."""
        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()

        npv = TN / (TN + FN) if (TN + FN) != 0 else 0
        ppv = TP / (TP + FP) if (TP + FP) != 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        print("\nClassification Report:\n", classification_report(y_true, y_pred))
        print(f"NPV: {npv:.2f}")
        print(f"PPV: {ppv:.2f}")
        print(f"Sensitivity: {sensitivity:.2f}")
        print(f"Specificity: {specificity:.2f}")

    # Train/evaluate each model
    for name, model in models.items():
        print(f"\nModel: {name}")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print_performance_metrics(y_test, y_pred)

    # 10-Fold Cross-Validation
    def evaluate_model_cross_validation_10fold(model, X_data, y_data, cv=10):
        ppvs, sensitivities, accuracies = [], [], []
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=99)

        for train_idx, test_idx in skf.split(X_data, y_data):
            X_cv_train, X_cv_test = X_data[train_idx], X_data[test_idx]
            y_cv_train, y_cv_test = y_data[train_idx], y_data[test_idx]

            model.fit(X_cv_train, y_cv_train)
            y_cv_pred = model.predict(X_cv_test)

            cm = confusion_matrix(y_cv_test, y_cv_pred)
            TN, FP, FN, TP = cm.ravel()

            ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0
            accuracy = (TP + TN) / (TP + TN + FP + FN)

            ppvs.append(ppv)
            sensitivities.append(sens)
            accuracies.append(accuracy)

        return {
            "PPV": np.mean(ppvs),
            "Sensitivity": np.mean(sensitivities),
            "Accuracy": np.mean(accuracies)
        }

    # Convert y_train to NumPy array for cross-validation
    y_train_np = y_train.to_numpy()

    cv_results_10fold = {}
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} with 10-Fold Cross Validation")
        results = evaluate_model_cross_validation_10fold(
            model, X_train_scaled, y_train_np, cv=10
        )
        cv_results_10fold[model_name] = results
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")

    print("\n10-Fold Cross Validation Results:")
    print(cv_results_10fold)


# -------------------------------------------------------------------------
# 3. Bayesian Optimization (CatBoost Example)
# -------------------------------------------------------------------------
def catboost_bayesian_opt_pipeline(csv_path: str) -> None:
    """
    Performs Bayesian optimization for CatBoost hyperparameters using a
    custom evaluation metric (weighted average of sensitivity and PPV).
    Saves logs to JSON and prints top 10 results.
    :param csv_path: path to the CSV file containing data
    """
    logging.basicConfig(
        filename='bayesopt.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # Load data
    df = pd.read_csv(csv_path)
    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    y_binary = (df.iloc[:, -1] >= 70).astype(int)

    X = df.iloc[:, :-1].drop(columns=['Year'], errors='ignore')
    logging.info("Data loaded for CatBoost Bayesian optimization.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=999, stratify=y_binary
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define evaluation function
    def catboost_eval(depth, iterations, learning_rate, l2_leaf_reg,
                      random_strength, bagging_temperature):
        # Cast to int
        depth_int = int(round(depth))
        iterations_int = int(round(iterations))

        params = {
            'depth': depth_int,
            'iterations': iterations_int,
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'random_strength': random_strength,
            'bagging_temperature': bagging_temperature,
            'random_state': 999,
            'eval_metric': 'AUC',
            'verbose': False,
            'thread_count': -1,
            'auto_class_weights': 'Balanced'
        }

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=999)
        cv_sensitivities = []
        cv_ppvs = []

        for train_idx, val_idx in skf.split(X_train_scaled, y_train):
            X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(X_cv_train, y_cv_train)

            preds = model.predict(X_cv_val)
            sens = recall_score(y_cv_val, preds, pos_label=1)
            ppv = precision_score(y_cv_val, preds, pos_label=1)

            cv_sensitivities.append(sens)
            cv_ppvs.append(ppv)

        avg_sensitivity = np.mean(cv_sensitivities)
        avg_ppv = np.mean(cv_ppvs)
        weighted_avg = (2 * avg_sensitivity + avg_ppv) / 3

        logging.info(
            f"depth={depth_int}, iterations={iterations_int}, "
            f"learning_rate={learning_rate:.4f}, l2_leaf_reg={l2_leaf_reg:.4f}, "
            f"random_strength={random_strength:.4f}, bagging_temperature={bagging_temperature:.4f}, "
            f"Sens={avg_sensitivity:.4f}, PPV={avg_ppv:.4f}, Weighted={weighted_avg:.4f}"
        )
        return weighted_avg

    # Bounds
    pbounds = {
        'depth': (3, 10),
        'iterations': (100, 1000),
        'learning_rate': (0.01, 0.3),
        'l2_leaf_reg': (1, 20),
        'random_strength': (0, 50),
        'bagging_temperature': (0, 10)
    }

    optimizer = BayesianOptimization(f=catboost_eval, pbounds=pbounds, random_state=999)

    # Loggers
    json_logger = JSONLogger(path="./logs.json")
    screen_logger = ScreenLogger()
    optimizer.subscribe(Events.OPTIMIZATION_STEP, json_logger)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, screen_logger)

    # Run optimization
    optimizer.maximize(init_points=5, n_iter=150)
    logging.info("Bayesian optimization completed.")

    # Report top 10
    top_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[:10]
    print("\nTop 10 Results:")
    for i, res in enumerate(top_results):
        logging.info(f"Top {i+1} result: {res}")
        print(f"Top {i+1} result: {res}")


# -------------------------------------------------------------------------
# 4. Final CatBoost Model Training & Evaluation
# -------------------------------------------------------------------------
def final_catboost_evaluation(csv_path: str, best_params: dict) -> None:
    """
    Trains a CatBoost model with the given best_params,
    evaluates on the test set, and prints confusion matrix & metrics.
    Also returns the trained model object for further usage.
    :param csv_path: path to the CSV file
    :param best_params: dictionary of best hyperparameters for CatBoost
    """
    df = pd.read_csv(csv_path)
    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    y_binary = (df.iloc[:, -1] >= 70).astype(int)

    X = df.iloc[:, :-1].drop(columns=['Year'], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=999, stratify=y_binary
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_best = CatBoostClassifier(**best_params)
    model_best.fit(X_train_scaled, y_train)

    # Evaluate
    y_test_pred = model_best.predict(X_test_scaled)

    test_sensitivity = recall_score(y_test, y_test_pred)
    test_ppv = precision_score(y_test, y_test_pred)
    print(f"\nTest set - Sensitivity: {test_sensitivity:.4f}, PPV: {test_ppv:.4f}")

    cm = confusion_matrix(y_test, y_test_pred)
    TN, FP, FN, TP = cm.ravel()
    npv = TN / (TN + FN) if (TN + FN) != 0 else 0
    ppv = TP / (TP + FP) if (TP + FP) != 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    print(f"Confusion Matrix:\n{cm}")
    print(f"NPV: {npv:.2f}")
    print(f"PPV: {ppv:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")

    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

    # Return the model if further usage is desired
    return model_best, X_train_scaled, y_train, X


# -------------------------------------------------------------------------
# 5. Feature Importances (CatBoost + SHAP)
# -------------------------------------------------------------------------
def feature_importance_analysis(model_best, X_train_scaled, y_train, original_features) -> None:
    """
    Prints CatBoost feature importances via LossFunctionChange and
    PredictionValuesChange, plus SHAP-based feature importance.
    :param model_best: trained CatBoost model
    :param X_train_scaled: scaled training features
    :param y_train: training labels
    :param original_features: list of original feature names
    """
    train_pool = Pool(X_train_scaled, label=y_train)

    # CatBoost: LossFunctionChange
    feature_importances_lfc = model_best.get_feature_importance(
        type='LossFunctionChange',
        data=train_pool
    )
    sorted_idx_lfc = np.argsort(feature_importances_lfc)[::-1]

    print("\nTop 20 Features by LossFunctionChange:")
    for idx in sorted_idx_lfc[:20]:
        print(f"{original_features[idx]}: {feature_importances_lfc[idx]:.5f}")

    # CatBoost: PredictionValuesChange
    feature_importances_pvc = model_best.get_feature_importance(
        type='PredictionValuesChange',
        data=train_pool
    )
    sorted_idx_pvc = np.argsort(feature_importances_pvc)[::-1]

    print("\nTop 20 Features by PredictionValuesChange:")
    for idx in sorted_idx_pvc[:20]:
        print(f"{original_features[idx]}: {feature_importances_pvc[idx]:.5f}")

    # SHAP
    explainer = shap.TreeExplainer(model_best)
    shap_values = explainer.shap_values(X_train_scaled)
    shap_values_abs_mean = np.abs(shap_values).mean(axis=0)
    sorted_idx_shap = np.argsort(shap_values_abs_mean)[::-1]

    print("\nTop 20 Features by SHAP Importance:")
    for i in range(20):
        feature_idx = sorted_idx_shap[i]
        print(f"{original_features[feature_idx]}: {shap_values_abs_mean[feature_idx]:.4f}")

    # SHAP Summary Plot
    shap.summary_plot(
        shap_values,
        X_train_scaled,
        plot_type="bar",
        max_display=20,
        feature_names=original_features
    )


# -------------------------------------------------------------------------
# 6. T-Test on Features
# -------------------------------------------------------------------------
def ttest_feature_significance(X_train_scaled, y_train, feature_names, top_n=200, out_csv='top_features_p_values.csv'):
    """
    Performs a t-test on each feature comparing class 0 and class 1,
    then saves the top_n features with smallest p-values to CSV.
    :param X_train_scaled: scaled training data (NumPy array or DataFrame)
    :param y_train: training labels
    :param feature_names: list of feature names
    :param top_n: number of top features to keep
    :param out_csv: where to save results
    """
    if isinstance(X_train_scaled, pd.DataFrame):
        X_train_scaled = X_train_scaled.values

    p_values_df = pd.DataFrame(columns=['Feature', 'P-value'])

    for i, feat in enumerate(feature_names):
        class_0_values = X_train_scaled[y_train == 0][:, i]
        class_1_values = X_train_scaled[y_train == 1][:, i]
        _, p_value = ttest_ind(class_0_values, class_1_values)
        p_values_df = p_values_df.append({'Feature': feat, 'P-value': p_value},
                                         ignore_index=True)

    # Sort and keep top_n
    top_features = p_values_df.sort_values(by='P-value').head(top_n)
    top_features.to_csv(out_csv, index=False)
    print(f"\nTop {top_n} Features by smallest p-value saved to {out_csv}:")
    print(top_features)


# -------------------------------------------------------------------------
# 7. Example Model with Top Features + SHAP Visualization
# -------------------------------------------------------------------------
def model_with_top_features(csv_path: str, top_features: list, random_state: int = 999) -> None:
    """
    Loads a dataset, selects a set of top features, trains a final CatBoost model,
    and visualizes SHAP values for specified features.
    :param csv_path: path to the CSV file
    :param top_features: list of feature names to keep
    :param random_state: random seed
    """
    # Load data
    df = pd.read_csv(csv_path)
    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    y_binary = (df.iloc[:, -1] >= 70).astype(int)
    X = df.iloc[:, :-1].drop(columns=['Year'], errors='ignore')

    # Keep only specified top features
    X = X[top_features]
    y = y_binary

    # Train/test split
    X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_unscaled)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=top_features)

    X_test_scaled = scaler.transform(X_test_unscaled)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=top_features)

    # Save scaler
    scaler_filename = f'scaler_random_state_{random_state}.joblib'
    dump(scaler, scaler_filename)
    print(f"Scaler saved as {scaler_filename}")

    # Example best params (adjust as needed)
    best_params = {
        'depth': 11,
        'iterations': 423,
        'l2_leaf_reg': 24.36,
        'learning_rate': 0.428,
        'eval_metric': 'AUC',
        'verbose': False,
        'thread_count': -1,
        'auto_class_weights': 'Balanced',
        'random_state': random_state
    }

    # Train CatBoost model
    model_final = CatBoostClassifier(**best_params)
    model_final.fit(X_train_scaled, y_train)

    # Save model
    model_filename = f'model_random_state_{random_state}.cbm'
    model_final.save_model(model_filename)
    print(f"Model saved as {model_filename}")

    # Evaluate on test set
    y_test_pred = model_final.predict(X_test_scaled)
    sens = recall_score(y_test, y_test_pred)
    ppv = precision_score(y_test, y_test_pred)
    print(f"Random State: {random_state}")
    print(f"Test Sensitivity: {sens:.4f}")
    print(f"Test PPV: {ppv:.4f}")

    # SHAP explanation
    explainer = shap.TreeExplainer(model_final)
    shap_values = explainer.shap_values(X_train_scaled)

    # Summary plot for all top features
    shap.summary_plot(shap_values, X_train_scaled, feature_names=top_features)

    # Example individual features to plot
    features_to_plot = ['densityPyr', 'bbod100200', 'Temperature', 'Aridity']
    for feature in features_to_plot:
        feature_idx = np.where(X_train_scaled.columns == feature)[0][0]
        feature_shap = shap_values[:, feature_idx]
        feature_vals = X_train_scaled[feature].values

        # Sort data for smoother plotting
        sorted_indices = np.argsort(feature_vals)
        sorted_feature_vals = feature_vals[sorted_indices]
        sorted_feature_shap = feature_shap[sorted_indices]

        # Apply Gaussian filter for smoothing
        smoothed_shap = gaussian_filter1d(sorted_feature_shap, sigma=2)

        plt.figure(figsize=(8, 5))
        plt.scatter(
            sorted_feature_vals,
            sorted_feature_shap,
            alpha=0.5,
            label='SHAP Values',
            color='blue'
        )
        plt.plot(sorted_feature_vals, smoothed_shap, 'r', label='Smoothed Curve')
        plt.title(f'SHAP Values for {feature}')
        plt.xlabel(f'Value of {feature}')
        plt.ylabel('SHAP Value')
        plt.legend()
        plt.grid(True)
        plt.show()


# -------------------------------------------------------------------------
# Example of how you might call these functions in a main block:
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. (Optional) KNN Imputation
    # knn_imputation("raw_data_with_nans.csv", "imputed_data.csv", n_neighbors=5)

    # 2. Preliminary Model Selection
    # model_selection_pipeline("updated_97770524pro.csv")

    # 3. Bayesian Optimization for CatBoost
    # catboost_bayesian_opt_pipeline("updated_97770524pro.csv")

    # model_best, X_train_scaled, y_train, X = final_catboost_evaluation(
    #     "imputed_97770524pro.csv", best_params_example
    # )

    # 4. Feature Importance (CatBoost + SHAP)
    # feature_importance_analysis(
    #     model_best,
    #     X_train_scaled,
    #     y_train,
    #     X.columns.tolist()
    # )

    # 5. T-Test
    # ttest_feature_significance(
    #     X_train_scaled, y_train, X.columns.tolist(),
    #     top_n=200,
    #     out_csv='0528top_200_features_p_values.csv'
    # )

    # 6. Model with Top Features + SHAP
    # top_12_features = [
    #     'Temperature',
    #     'densityPyr',
    #     'ocs_030',
    #     'Rainy Precipitation',
    #     'underground level',
    #     'bbod100200',
    #     'water source.1',
    #     'cfvo100200',
    #     'Dry Precipitation',
    #     'Aridity',
    #     'Dry Temperature',
    #     'standards'
    # ]
    # model_with_top_features("imputed_97770524pro.csv", top_12_features, random_state=999)

    print("Module loaded. Uncomment desired function calls in the __main__ block to execute.")
