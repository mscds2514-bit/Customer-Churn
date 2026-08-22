import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel("data/Telco_customer_churn.xlsx")

df.columns = df.columns.str.strip()

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==========================================
# 2. REMOVE COLUMNS NOT NEEDED FOR PREDICTION
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

# Remove only columns that exist
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
# 12. SELECT ONE CUSTOMER FOR PREDICTION
# ==========================================

customer = X_test.iloc[[0]]


# ==========================================
# 13. MAKE PREDICTION
# ==========================================

prediction = model.predict(customer)[0]

probability = model.predict_proba(customer)[0][1]


# ==========================================
# 14. DISPLAY CUSTOMER DETAILS
# ==========================================

print("\n====================================")
print("CUSTOMER DETAILS")
print("====================================")

print(customer.to_string(index=False))


# ==========================================
# 15. DISPLAY PREDICTION
# ==========================================

print("\n====================================")
print("CUSTOMER CHURN PREDICTION")
print("====================================")

if prediction == 1:
    print("Prediction: Customer is likely to CHURN")
else:
    print("Prediction: Customer is likely to STAY")

print(
    "Churn probability:",
    round(probability * 100, 2),
    "%"
)

print("====================================")