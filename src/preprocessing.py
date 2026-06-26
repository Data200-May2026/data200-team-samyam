import pandas as pd

df = pd.read_csv("/content/ai_job_trends_dataset.csv")
df.drop_duplicates(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
df.to_csv("/content/cleaned_data.csv", index=False)

print(df.info())
print(df.describe())
