import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

df = pd.read_csv("/content/cleaned_data.csv")

X = df[[
    "Experience Required (Years)",
    "Remote Work Ratio (%)",
    "Automation Risk (%)",
    "Gender Diversity (%)"
]]

y = df["Projected Openings (2030)"]

selector = SelectKBest(score_func=f_regression,k='all')
selector.fit(X,y)

for score,col in zip(selector.scores_,X.columns):
    print(col,score)
