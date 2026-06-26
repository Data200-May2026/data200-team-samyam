import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/content/cleaned_data.csv")

plt.figure(figsize=(10,6))
sns.histplot(df["Automation Risk (%)"], bins=30)
plt.savefig("automation_risk_hist.png")

plt.figure(figsize=(10,6))
sns.boxplot(x=df["Median Salary (USD)"])
plt.savefig("salary_boxplot.png")

numeric = df.select_dtypes(include='number')

plt.figure(figsize=(12,8))
sns.heatmap(numeric.corr(), annot=True)
plt.savefig("correlation_heatmap.png")
