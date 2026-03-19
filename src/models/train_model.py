import sklearn
import pandas as pd
from sklearn import ensemble
from sklearn.impute import SimpleImputer
import joblib
import numpy as np

print(joblib.__version__)

def coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # include both object and string dtypes (pandas 2 + 3 safe)
    for c in df.select_dtypes(include=["object", "string"]).columns:
        s = df[c].astype(str)
        s = s.str.replace("Â", "", regex=False)
        s = s.str.replace("\u00a0", "", regex=False)  # non-breaking space
        s = s.str.replace(" ", "", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce")

    return df


# ---------------------
# Load data
# ---------------------
X_train = pd.read_csv('data/preprocessed/X_train.csv', low_memory=False)
X_test  = pd.read_csv('data/preprocessed/X_test.csv',  low_memory=False)
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test  = pd.read_csv('data/preprocessed/y_test.csv')

y_train = np.ravel(y_train)
y_test  = np.ravel(y_test)

# ---------------------
# Fix encoding artifacts
# ---------------------
X_train = coerce_numeric_frame(X_train)
X_test  = coerce_numeric_frame(X_test)

# ---------------------
# Drop columns that are completely NaN
# ---------------------
all_nan_cols = X_train.columns[X_train.isna().all()]

if len(all_nan_cols) > 0:
    print("Dropping fully empty columns:", list(all_nan_cols))
    X_train = X_train.drop(columns=all_nan_cols)
    X_test  = X_test.drop(columns=all_nan_cols)

# ---------------------
# Impute remaining NaNs
# ---------------------
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_train.columns
)

# ---------------------
# Train model
# ---------------------
rf_classifier = ensemble.RandomForestClassifier(
        n_jobs=-1,
        n_estimators=200,
        criterion="entropy"
        )
rf_classifier.fit(X_train, y_train)

# ---------------------
# Save model
# ---------------------
model_filename = './models/trained_model.joblib'
joblib.dump(rf_classifier, model_filename)

print("✅ Model trained and saved successfully.")
