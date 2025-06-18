import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your final fused feature dataset
df = pd.read_csv('C:/Users/s233183/OneDrive - Danmarks Tekniske Universitet/Desktop/Special-Course-Parkinson-Machine-Learning/GaitProject_NRT/results/fused_features.csv')

# Select features and target
feature_cols = [col for col in df.columns if col not in ['Start', 'End', 'Subject', 'Muscle', 'Group']]
X = df[feature_cols]
y = df['Group'].map({'HC': 0, 'PD': 1})  # encode target as 0 = HC, 1 = PD

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42)
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
results = {'Model': [], 'Accuracy Mean': [], 'Accuracy Std': [], 'F1 Score Mean': []}

# Run CV for each model
for name, model in models.items():
    acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')

    results['Model'].append(name)
    results['Accuracy Mean'].append(np.mean(acc_scores))
    results['Accuracy Std'].append(np.std(acc_scores))
    results['F1 Score Mean'].append(np.mean(f1_scores))

# Create result DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plot results
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy Mean', data=results_df, palette='viridis')
plt.title('5-Fold Cross-Validated Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()

# Optional: also plot F1 scores
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='F1 Score Mean', data=results_df, palette='magma')
plt.title('5-Fold Cross-Validated F1 Score')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()

# Pretty table visualization using seaborn heatmap
plt.figure(figsize=(8, 2))
sns.heatmap(results_df[['Accuracy Mean', 'F1 Score Mean']].T, annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=results_df['Model'].values)
plt.title('Model Performance Summary')
plt.yticks(rotation=0)
plt.show()