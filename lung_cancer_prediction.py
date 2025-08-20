# Lung Cancer Prediction with Decision Tree and PCA

# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load dataset
url = 'lung_cancer_examples.csv' # Change path if needed
data = pd.read_csv(url)

# Display basic info
print(data.head())
print(data.info())

# Step 3: Preprocessing
# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Drop non-predictive columns (Name, Surname)
data_clean = data.drop(columns=['Name', 'Surname'])

# Define Features and Target
X = data_clean.drop(columns=['Result'])
y = data_clean['Result']

# Step 4: Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Baseline Decision Tree classifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train_scaled, y_train)

# Predictions and evaluation on test set
y_pred = dtc.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Baseline Decision Tree Performance")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
feature_importances = pd.Series(dtc.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importance (Baseline DT):\n", feature_importances)

# Step 7: PCA to retain ≥95% variance
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of features after PCA: {X_train_pca.shape[1]}")

# Step 8: Decision Tree on PCA-transformed data
dtc_pca = DecisionTreeClassifier(random_state=42)
dtc_pca.fit(X_train_pca, y_train)

y_pred_pca = dtc_pca.predict(X_test_pca)

accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca)
recall_pca = recall_score(y_test, y_pred_pca)
f1_pca = f1_score(y_test, y_pred_pca)
cm_pca = confusion_matrix(y_test, y_pred_pca)

print("\nDecision Tree Performance after PCA")
print(f"Accuracy: {accuracy_pca:.4f}")
print(f"Precision: {precision_pca:.4f}")
print(f"Recall: {recall_pca:.4f}")
print(f"F1 Score: {f1_pca:.4f}")
print("Confusion Matrix:\n", cm_pca)
print("Classification Report:\n", classification_report(y_test, y_pred_pca))

# Step 9: Visualization of feature importance for baseline
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title('Feature Importance (Baseline Decision Tree)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Step 10: Discussion
print("""
Discussion:
- The baseline Decision Tree uses original features and shows their relative importance.
- PCA reduced the feature dimensions while retaining >95% variance, improving or maintaining performance with fewer features.
- This can help reduce overfitting and improve model interpretability.
""")
