import pandas as pd
import statsmodels.api as sm

df = pd.read_csv(
    "encoded_data.csv"
)

target = "Projected Openings (2030)"

X = df.drop(
    columns=[target]
)

y = df[target]

X = sm.add_constant(X)

model = sm.OLS(
    y,
    X
).fit()

print(model.summary())

with open(
    "../reports/ols_summary.txt",
    "w"
) as f:
    f.write(
        model.summary().as_text()
    )

coef = pd.DataFrame({
    "Feature":model.params.index,
    "Coefficient":model.params.values,
    "P_Value":model.pvalues.values
})

coef.to_csv(
    "../reports/ols_coefficients.csv",
    index=False
)
