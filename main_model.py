import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

#Data Gathering

print("\n Loading dataset...")
data = pd.read_csv("student_data.csv")
print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.\n")

# Drop irrelevant columns
drop_cols = [
    "Application mode", "Application order", "Daytime/evening attendance",
    "Previous qualification (grade)", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Admission grade"
]
data.drop(columns=drop_cols, inplace=True, errors='ignore')


#Data Cleaning

print(" Handling missing values...")

num_cols = data.select_dtypes(include=['float64', 'int64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

# Visualizing missing values before  cleaning the dataset
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap="Reds")
plt.title("Missing Values Before Cleaning")
plt.show()

# Handle missing values
data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

print("Missing values handled successfully.\n")

# Visualizing missing values AFTER cleaning

plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap="Greens")
plt.title("Missing Values After Cleaning")
plt.show()


#Code for cleaning outliers

print(" Detecting and handling outliers with IQR method...")

# Outliers Before
plt.figure(figsize=(14, 7))
sns.boxplot(data=data[num_cols])
plt.title("Outliers Before Capping")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Capping outliers
for col in num_cols:
    q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    data[col] = np.where(data[col] < lower, lower,
                         np.where(data[col] > upper, upper, data[col]))

print("Outliers capped successfully.\n")

# Outliers After
plt.figure(figsize=(14, 7))
sns.boxplot(data=data[num_cols])
plt.title("Outliers After Capping")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#Descriptive statistics

print(" Descriptive Statistics Summary:")
print(data.describe().round(2))
print("\n")

# Distribution of student target classes
if "Target" in data.columns:
    plt.figure(figsize=(6, 6))
    data["Target"].value_counts().plot.pie(
        autopct="%1.1f%%",
        labels=["Dropout", "Enrolled", "Graduate"],
        colors=["red", "orange", "green"]
    )
    plt.title("Distribution of Student Academic Outcomes")
    plt.ylabel("")
    plt.show()

#Feature encoding and scaling


print(" Encoding categorical variables...")

label_encoders = {}
for col in cat_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

print("Encoding complete.\n")

# Splitting X and y
X = data.drop("Target", axis=1)
y = data["Target"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#Code to split the dataset and Train the dataset

print(" Splitting dataset into train & test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=109, stratify=y
)

print(f"Training size: {X_train.shape}, Testing size: {X_test.shape}\n")


#Baseline Model Training

print(" Training baseline Random Forest model...")

model = RandomForestClassifier(random_state=109)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n Baseline Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))


#Model Tuning(Grid Search)

print("\n Performing Hyperparameter Tuning...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\n Best Parameters Found:")
print(grid.best_params_)


# Evaluate tuned model
y_pred_tuned = best_model.predict(X_test)

print("\n Tuned Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_tuned), 4))
print(classification_report(y_test, y_pred_tuned))


#Confusion Matrix 

cm = confusion_matrix(y_test, y_pred_tuned)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — Tuned Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


#Feature Importance

importances = pd.Series(best_model.feature_importances_, index=X.columns)
top_features = importances.nlargest(10)

print("\n Top 10 Most Important Features:\n", top_features)

plt.figure(figsize=(8, 6))
top_features.plot(kind='barh', color='teal')
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()


#Save Model and Encoders

print("\n Saving trained model & preprocessors...")

joblib.dump(best_model, "student_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print(" All model files saved successfully!")
