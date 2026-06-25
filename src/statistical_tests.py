import pandas as pd
from scipy.stats import ttest_ind, f_oneway

df = pd.read_csv("/content/cleaned_data.csv")

# Inspect the unique values and counts of 'Job Status'
print("Unique values and counts in 'Job Status' column:")
print(df['Job Status'].value_counts())

results = []
increasing = df[
    df["Job Status"] == "Increasing"
]["Median Salary (USD)"]

declining = df[
    df["Job Status"] == "Declining"
]["Median Salary (USD)"]

t_stat, p_value = ttest_ind(
    increasing,
    declining,
    equal_var=False
)

results.append({
    "Test":"T-Test",
    "Comparison":"Increasing vs Declining Salary",
    "Statistic":t_stat,
    "P_Value":p_value
})
low = df[
    df["AI Impact Level"] == "Low"
]["Median Salary (USD)"]

moderate = df[
    df["AI Impact Level"] == "Moderate"
]["Median Salary (USD)"]

high = df[
    df["AI Impact Level"] == "High"
]["Median Salary (USD)"]

f_stat, p_value = f_oneway(
    low,
    moderate,
    high
)

results.append({
    "Test":"ANOVA",
    "Comparison":"AI Impact Groups",
    "Statistic":f_stat,
    "P_Value":p_value
})

output = pd.DataFrame(results)

output.to_csv(
    "/content/statistical_test_results.csv", # Changed path to /content/
    index=False
)

print(output)
