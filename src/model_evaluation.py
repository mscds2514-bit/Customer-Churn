import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel("data/Telco_customer_churn.xlsx")

df.columns = df.columns.str.strip()

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==========================================
# 2. REMOVE COLUMNS NOT NEEDED FOR MODEL
# ==========================================

columns_to_remove = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Lat Long",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason"
]

columns_to_remove = [
    column for column in columns_to_remove
    if column in df.columns
]

df = df.drop(columns=columns_to_remove)


# ==========================================
# 3. CONVERT TOTAL CHARGES TO NUMERIC
# ==========================================

if "Total Charges" in df.columns:

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )


# ==========================================
# 4. DEFINE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Churn Label"])

y = df["Churn Label"].map({
    "Yes": 1,
    "No": 0
})


# ==========================================
# 5. IDENTIFY FEATURE TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ==========================================
# 6. NUMERIC PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ==========================================
# 7. CATEGORICAL PREPROCESSING
# ==========================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ==========================================
# 8. COMBINE PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==========================================
# 9. CREATE MACHINE LEARNING MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])


# ==========================================
# 10. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 11. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)


# ==========================================
# 12. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 13. CALCULATE EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================
# 14. DISPLAY MODEL EVALUATION
# ==========================================

print("\n====================================")
print("MODEL EVALUATION")
print("====================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ==========================================
# 15. CLASSIFICATION REPORT
# ==========================================

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"]
    )
)


# ==========================================
# 16. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n====================================")
print("CONFUSION MATRIX")
print("====================================")

print(cm)


# ==========================================
# 17. CONFUSION MATRIX VISUALIZATION
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()

plt.title("Customer Churn - Confusion Matrix")

plt.tight_layout()

plt.show()


# ==========================================
# 18. ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Customer Churn")

plt.legend()

plt.tight_layout()

plt.show()