import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


def coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        s = df[c].astype(str)
        s = s.str.replace("Â", "", regex=False)
        s = s.str.replace("\u00a0", "", regex=False)
        s = s.str.replace(" ", "", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce")
    return df


X_test = pd.read_csv('data/preprocessed/X_test.csv', low_memory=False)
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_test = np.ravel(y_test)

# Apply same preprocessing as in training
X_test = coerce_numeric_frame(X_test)


def main(repo_path):
    model = load(repo_path / "models/trained_model.joblib")

    # Align X_test columns to those seen during training
    train_features = model.feature_names_in_
    X_test_aligned = X_test.reindex(columns=train_features)

    # Impute with median (same strategy as training)
    imputer = SimpleImputer(strategy="median")
    X_test_imputed = pd.DataFrame(
        imputer.fit_transform(X_test_aligned),
        columns=train_features
    )

    predictions = model.predict(X_test_imputed)
    accuracy = accuracy_score(y_test, predictions)
    metrics = {"accuracy": accuracy}
    accuracy_path = repo_path / "metrics/accuracy.json"
    accuracy_path.write_text(json.dumps(metrics))


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)
