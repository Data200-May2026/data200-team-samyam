import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("cleaned_data.csv")

target = "Projected Openings (2030)"

X = df.drop(columns=[target])

y = df[target]

categorical = X.select_dtypes(
    include="object"
).columns

numeric = X.select_dtypes(
    exclude="object"
).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical
        ),
        (
            "num",
            "passthrough",
            numeric
        )
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

pipeline.fit(
    X_train,
    y_train
)

pred = pipeline.predict(
    X_test
)

metrics = pd.DataFrame({
    "Metric":[
        "MAE",
        "RMSE",
        "R2"
    ],
    "Value":[
        mean_absolute_error(
            y_test,
            pred
        ),
        mean_squared_error(
            y_test,
            pred
        ) ** 0.5,
        r2_score(
            y_test,
            pred
        )
    ]
})

metrics.to_csv(
    "../reports/regression_metrics.csv",
    index=False
)

joblib.dump(
    pipeline,
    "../models/linear_regression.pkl"
)

print(metrics)
