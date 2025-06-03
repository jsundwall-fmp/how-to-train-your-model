# python -m pip install pandas xgboost matplotlib scikit-learn joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
import optuna


def objective(trial):
  params = {
    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
    "max_depth": trial.suggest_int("max_depth", 2, 10),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
  }

  model = XGBClassifier(**params)
  model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
  y_pred = model.predict(X_val)
  y_prob = model.predict_proba(X_val)[:, 1]
  roc_auc = roc_auc_score(y_val, y_prob)
  return roc_auc

# Import necessary libraries

# === CONFIGURATION ===
CSV_PATH = "training_data.csv"
TARGET_COLUMN = "label_3"
RANDOM_SEED = 42
SAVE_METRICS_TO = "run_metrics.json"
SHAP_SAMPLE_SIZE = 1000
EXPERIMENT_NAME = "hint_lover_xgb"

print("🔍 Loading and preprocessing data...")
# === LOAD AND CLEAN DATA ===
df = pd.read_csv(CSV_PATH)
df.columns = [col.lower() for col in df.columns]
df.fillna(0, inplace=True)

# Split features and labels
EXCLUDED_FEATURES = ["'open_chat_from_quick_merge_last_1_days'", "'previous_page_last_7_days'", "'finish_last_3_days'", "'display_expanded_explainer_last_7_days'", "'all_hints_category_filter_no_hints_last_1_days'", "'full_profile_no_hints_last_7_days'", "'view_source_from_quick_merge_last_3_days'", "'drawer_error_last_7_days'", "'minimise_explainer_last_3_days'", "'drawer_no_hints_last_7_days'", "'next_page_last_3_days'", "'explore_content_shown_last_7_days'", "'message_tree_owner_from_quick_merge_last_3_days'", "'all_hints_category_filter_no_hints_last_7_days'", "'all_hints_category_filter_last_7_days'", "'explore_content_expanded_last_7_days'", "'back_to_tree_last_7_days'", "'open_chat_from_quick_merge_last_7_days'", "'explore_content_shown_last_1_days'", "'full_profile_no_hints_last_3_days'", "'francis_frith_opened_last_30_days'", "'explore_content_expanded_last_3_days'", "'back_to_tree_last_30_days'", "'hint_card_more_menu_button_click_last_7_days'", "'back_to_tree_last_3_days'", "'hint_card_more_menu_button_click_last_3_days'", "'all_hints_error_last_3_days'", "'expand_explainer_last_30_days'", "'rejected_hint_error_last_30_days'", "'all_hints_category_filter_no_hints_last_30_days'", "'explore_content_expanded_last_30_days'", "'rejected_hint_error_last_7_days'", "'all_hints_error_last_7_days'", "'expand_explainer_last_1_days'", "'full_profile_no_hints_last_30_days'", "'finish_last_30_days'", "'hint_saved_last_3_days'", "'saved_hint_error_last_30_days'", "'blue_plaque_opened_last_30_days'", "'rejected_hint_error_last_3_days'", "'all_hints_sort_by_last_30_days'", "'all_hints_clear_filter_last_30_days'", "'all_hints_clear_filter_last_1_days'", "'drawer_hints_menu_button_click_last_30_days'", "'expand_explainer_last_3_days'", "'drawer_hints_menu_button_click_last_3_days'", "'drawer_hints_menu_button_click_last_7_days'", "'hint_card_more_menu_button_click_last_30_days'", "'explore_opened_last_30_days'", "'all_hints_sort_by_last_3_days'", "'display_expanded_explainer_last_3_days'", "'all_hints_clear_filter_last_7_days'", "'all_hints_error_last_30_days'", "'expand_explainer_last_7_days'", "'explore_content_shown_last_30_days'", "'open_chat_from_quick_merge_last_30_days'", "'drawer_error_last_3_days'", "'previous_page_last_3_days'", "'all_hints_clear_filter_last_3_days'", "'saved_hint_error_last_7_days'", "'hints_for_this_tree_last_3_days'", "'next_page_last_7_days'", "'message_tree_owner_from_quick_merge_last_7_days'", "'all_hints_category_filter_last_3_days'", "'saved_hint_error_last_1_days'", "'message_tree_owner_from_quick_merge_last_30_days'", "'open_chat_from_quick_merge_last_3_days'", "'francis_frith_opened_last_7_days'", "'explore_content_shown_last_3_days'", "'previous_page_last_30_days'", "'saved_hint_error_last_3_days'", "'all_hints_sort_by_last_7_days'", "'all_hints_category_filter_no_hints_last_3_days'"]
NON_FEATURE_COLUMNS = ["event_id", "user_id", "derived_tstamp", "has_managed_hint_next_1_days", "has_managed_hint_next_3_days", "has_managed_hint_next_7_days", "label_1", "label_3", "label_7"]
feature_columns = [col for col in df.columns if col not in NON_FEATURE_COLUMNS and col not in EXCLUDED_FEATURES and df[col].dtype != "object"]

X = df[feature_columns]
y = df[TARGET_COLUMN]

print(f"✅ Loaded {len(df)} rows with {len(feature_columns)} features")

# === SPLIT INTO TRAIN/VAL/TEST ===
print("📊 Splitting data into train/validation/test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp)

print(f"📁 Training set: {len(X_train)} rows")
print(f"📁 Validation set: {len(X_val)} rows")
print(f"📁 Test set: {len(X_test)} rows")


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=10, n_jobs=1)
print("🔬 Starting hyperparameter optimization...")


# === TRAIN MODEL ===
print("🚀 Training XGBoost model...")
model = XGBClassifier(
  **study.best_params,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)
print("✅ Model training complete.")

print("📈 Evaluating model on validation set...")
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, y_prob)
cm = confusion_matrix(y_val, y_pred).tolist()
cr = classification_report(y_val, y_pred, output_dict=True)

# Log metrics
print("roc_auc", roc_auc)
print("confusion_matrix", cm)
print('classification_report', cr)
print("f1_class_0", cr["0"]["f1-score"])
print("f1_class_1", cr["1"]["f1-score"])
print("accuracy", cr["accuracy"])

# Save confusion matrix as artifact
plt.figure()
plt.imshow(confusion_matrix(y_val, y_pred), cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# Save and log model
model_path = "hint_lover_model.pkl"
joblib.dump(model, model_path)

# Log feature importances
fig, ax = plt.subplots()
ax.hist(model.feature_importances_, bins=50)
ax.axvline(0.002, color='green', linestyle='--', label="0.002")
ax.axvline(0.005, color='red', linestyle='--', label="0.005")
ax.set_title("Feature Importance Distribution")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature Count")
ax.legend()
fig.tight_layout()
fig.savefig("feature_importance.png")

# Save ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.grid()
plt.tight_layout()
plt.savefig("roc_curve.png")

# Save PR Curve
precision, recall, _ = precision_recall_curve(y_val, y_prob)
ap_score = average_precision_score(y_val, y_prob)
plt.figure()
plt.plot(recall, precision, label=f"AP = {ap_score:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.tight_layout()
plt.savefig("pr_curve.png")


# SHAP Values
explainer = shap.Explainer(model)
X_val_sample = X_val.sample(SHAP_SAMPLE_SIZE, random_state=RANDOM_SEED)
shap_values = explainer(X_val_sample)

# Identify low importance features
shap_array = shap_values.values
shap_array = np.nan_to_num(shap_array, nan=0.0)
mean_abs_shap_values = np.abs(shap_array).mean(axis=0)

shap_importance = pd.Series(mean_abs_shap_values, index=X_val_sample.columns, name="Mean |SHAP|")
shap_importance.sort_values(ascending=False, inplace=True)

print(shap_importance)

threshold = 0.002
low_importance_features = shap_importance[shap_importance < threshold].index.tolist()

print(f"Low importance features (mean SHAP < 0.002): {low_importance_features}")
