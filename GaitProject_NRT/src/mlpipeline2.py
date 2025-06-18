import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the fused dataset
df = pd.read_csv('C:/Users/s233183/OneDrive - Danmarks Tekniske Universitet/Desktop/Special-Course-Parkinson-Machine-Learning/GaitProject_NRT/results/fused_features.csv')
feature_cols = [c for c in df.columns if c not in ['Start', 'End', 'Subject', 'Muscle', 'Group']]
X = df[feature_cols]
y = df['Group'].map({'HC': 0, 'PD': 1})

# Stratified CV setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define pipelines
pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='linear', probability=True, random_state=42))
    ]),
}

# Scoring metrics
scoring = ['accuracy', 'f1', 'roc_auc']

# Cross-validate
cv_results = {}
for name, pipe in pipelines.items():
    results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False)
    cv_results[name] = {
        'Accuracy Mean': np.mean(results['test_accuracy']),
        'Accuracy Std':  np.std(results['test_accuracy']),
        'F1 Mean':       np.mean(results['test_f1']),
        'ROC AUC Mean':  np.mean(results['test_roc_auc'])
    }

results_df = pd.DataFrame(cv_results).T
print(results_df)

# Plot ROC curve (mean across folds) for the best model (Random Forest)
best_pipe = pipelines['Random Forest']
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []
for train_idx, test_idx in cv.split(X, y):
    best_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    probas_ = best_pipe.predict_proba(X.iloc[test_idx])[:, 1]
    fpr, tpr, _ = roc_curve(y.iloc[test_idx], probas_)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    tprs.append(interp_tpr)
    aucs.append(roc_auc_score(y.iloc[test_idx], probas_))

plt.figure(figsize=(8, 5))
mean_tpr = np.mean(tprs, axis=0)
plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {np.mean(aucs):.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Random Forest ROC Curve (5-Fold CV)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrix via cross_val_predict
y_pred = cross_val_predict(best_pipe, X, y, cv=cv)
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['HC', 'PD'], yticklabels=['HC', 'PD'])
plt.title('Random Forest Confusion Matrix (5-Fold CV)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance from Random Forest
best_pipe.fit(X, y)
importances = best_pipe.named_steps['clf'].feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10])
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Optional: Hyperparameter tuning (example for RF)
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 5, 10]
}
grid_search = GridSearchCV(pipelines['Random Forest'], param_grid, cv=cv, scoring='roc_auc')
grid_search.fit(X, y)
print("Best RF params:", grid_search.best_params_)
print("Best RF AUC:", grid_search.best_score_)