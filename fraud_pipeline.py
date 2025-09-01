import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import joblib
df = pd.read_csv("creditcard.csv")
print("Data shape:", df.shape)
print("Fraud class distribution:\n", df['Class'].value_counts())
plt.figure(figsize=(6,4))
df['Class'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Fraud vs Non-Fraud Transaction Counts')
plt.xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['Time','Amount']] = scaler.fit_transform(X_scaled[['Time','Amount']])
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)
print("\nTrain class distribution:\n", y_train.value_counts())
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("\nUsed SMOTE. New train distribution:\n", pd.Series(y_res).value_counts())
except Exception as e:
    print("\nSMOTE not available, using Random oversampling. Reason:", e)
    train = pd.concat([X_train, y_train], axis=1)
    fraud = train[train['Class']==1]
    non_fraud = train[train['Class']==0]
    fraud_upsampled = resample(
        fraud, replace=True, n_samples=len(non_fraud), random_state=42
    )
    upsampled = pd.concat([non_fraud, fraud_upsampled])
    X_res = upsampled.drop('Class', axis=1)
    y_res = upsampled['Class']
    print("After upsampling train distribution:\n", y_res.value_counts())

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results[name] = {
        "model": model,
        "classification_report": report,
        "confusion_matrix": conf,
        "roc_auc": roc_auc,
        "pr_auc": avg_prec,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    print(f"\n=== {name} ===")
    print("Confusion matrix:\n", conf)
    print("Precision: {:.4f}  Recall: {:.4f}  F1: {:.4f}".format(prec, rec, f1))
    print("ROC-AUC: {:.4f}  PR-AUC: {:.4f}".format(roc_auc, avg_prec))
best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
print("\nBest model by F1:", best_model_name)

best_model = results[best_model_name]['model']
report_df = pd.DataFrame(results[best_model_name]['classification_report']).transpose()
print("\nClassification Report:\n", report_df)
y_test_arr = y_test.values
y_proba_best = results[best_model_name]['y_proba']
fpr, tpr, _ = roc_curve(y_test_arr, y_proba_best)
precision_curve, recall_curve, _ = precision_recall_curve(y_test_arr, y_proba_best)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title(f"ROC Curve ({best_model_name})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.subplot(1,2,2)
plt.plot(recall_curve, precision_curve)
plt.title(f"Precision-Recall Curve ({best_model_name})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()
joblib.dump(best_model, f"best_fraud_model_{best_model_name}.joblib")
print(f"\nSaved best model to: best_fraud_model_{best_model_name}.joblib")
test_df = X_test.copy()
test_df['true_Class'] = y_test.values
test_df['fraud_proba'] = results[best_model_name]['y_proba']
test_df['predicted_Class'] = results[best_model_name]['y_pred']
top15 = test_df.sort_values('fraud_proba', ascending=False).head(15)
print("\nTop 15 transactions predicted as fraud:")
print(top15)