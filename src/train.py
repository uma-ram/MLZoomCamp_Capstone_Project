import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


def train_model():
    # Load data
    df = pd.read_csv("data/train.csv")

    X = df.drop("y", axis=1)
    y = df["y"].map({"no": 0, "yes": 1})

    # Feature types
    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    # Final tuned XGBoost model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42
            ))
        ]
    )

    # Train
    model.fit(X, y)

    # Save model
    joblib.dump(model, "models/model.bin")
    print("Model saved to models/model.bin")


if __name__ == "__main__":
    train_model()
