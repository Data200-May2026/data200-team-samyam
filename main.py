
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
 
import statsmodels.api as sm
from scipy.stats import f_oneway, ttest_ind
 
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Trends – Model Dashboard",
    page_icon="📊",
    layout="wide",
)
 
DATA_URL = (
    "https://raw.githubusercontent.com/Data200-May2026/"
    "data200-team-samyam/main/data/raw/ai_job_trends_dataset.csv"
)
 
# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    df = pd.read_csv(DATA_URL)
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df
 
df = load_data()
 
TARGET = "Projected Openings (2030)"
NUMERIC_FEATURES = [
    "Experience Required (Years)",
    "Remote Work Ratio (%)",
    "Automation Risk (%)",
    "Gender Diversity (%)",
]
 
# ── Train model (cached so it only runs once) ─────────────────────────────────
@st.cache_data(show_spinner="Training model …")
def train_model(data):
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
 
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()
 
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
 
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R²": r2_score(y_test, pred),
    }
    return metrics, y_test.values, pred
 
 
@st.cache_data(show_spinner="Running OLS …")
def run_ols(data):
    le = LabelEncoder()
    encoded = data.copy()
    for col in encoded.select_dtypes(include=["object", "string"]).columns:
        encoded[col] = le.fit_transform(encoded[col])
    X = encoded.drop(columns=[TARGET])
    y = encoded[TARGET]
    X_c = sm.add_constant(X)
    model = sm.OLS(y, X_c).fit()
    coef_df = pd.DataFrame({
        "Feature": model.params.index,
        "Coefficient": model.params.values,
        "Std Error": model.bse.values,
        "P-Value": model.pvalues.values,
    })
    summary_stats = {
        "R²": round(model.rsquared, 4),
        "Adj. R²": round(model.rsquared_adj, 4),
        "F-Statistic": round(model.fvalue, 4),
        "Prob (F-stat)": f"{model.f_pvalue:.2e}",
        "AIC": round(model.aic, 2),
        "BIC": round(model.bic, 2),
        "Observations": int(model.nobs),
    }
    return coef_df, summary_stats
 
 
metrics, y_test, pred = train_model(df)
coef_df, ols_summary = run_ols(df)
 
# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 AI Job Trends – Model Results Dashboard")
st.caption("DATA 200 · Team Samyam · Westcliff University")
st.markdown("---")
 
# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "🔍 EDA",
    "🧪 Hypothesis Tests",
    "⚙️ Feature Selection",
    "📈 Model Results",
])
 
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Industries", df["Industry"].nunique())
    c4.metric("Locations", df["Location"].nunique())
 
    st.markdown("#### Sample Data (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)
 
    st.markdown("#### Descriptive Statistics")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
 
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Exploratory Data Analysis")
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown("**Salary Distribution**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df["Median Salary (USD)"], bins=30, kde=True, ax=ax, color="#4C72B0")
        ax.set_xlabel("Median Salary (USD)")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
    with col_b:
        st.markdown("**Automation Risk Distribution**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df["Automation Risk (%)"], bins=30, kde=True, ax=ax, color="#DD8452")
        ax.set_xlabel("Automation Risk (%)")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
    col_c, col_d = st.columns(2)
 
    with col_c:
        st.markdown("**Projected Openings (2030) by AI Impact Level**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(data=df, x="AI Impact Level", y=TARGET,
                    order=["Low", "Moderate", "High"], ax=ax, palette="Set2")
        ax.set_ylabel("Projected Openings (2030)")
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
    with col_d:
        st.markdown("**Job Count by Industry**")
        counts = df["Industry"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts.plot(kind="bar", ax=ax, color="#55A868")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=35)
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
    st.markdown("**Correlation Heatmap (Numeric Features)**")
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close()
 
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – HYPOTHESIS TESTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Hypothesis Tests")
 
    # ANOVA ───────────────────────────────────────────────────────────────────
    st.markdown("### One-Way ANOVA")
    st.markdown(
        "**H₀:** Mean salary is the same across all AI Impact Levels.  \n"
        "**H₁:** At least one group has a different mean salary."
    )
    low    = df[df["AI Impact Level"] == "Low"]["Median Salary (USD)"]
    mod    = df[df["AI Impact Level"] == "Moderate"]["Median Salary (USD)"]
    high   = df[df["AI Impact Level"] == "High"]["Median Salary (USD)"]
    f_stat, p_anova = f_oneway(low, mod, high)
 
    ca, cb, cc = st.columns(3)
    ca.metric("F-Statistic", f"{f_stat:.4f}")
    cb.metric("P-Value", f"{p_anova:.4f}")
    cc.metric(
        "Result",
        "Reject H₀ ✅" if p_anova < 0.05 else "Fail to Reject H₀ ❌",
        delta=None,
    )
    if p_anova < 0.05:
        st.success("There is a statistically significant difference in median salary across AI impact levels (p < 0.05).")
    else:
        st.info("No statistically significant difference found (p ≥ 0.05).")
 
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.boxplot(data=df, x="AI Impact Level", y="Median Salary (USD)",
                order=["Low", "Moderate", "High"], ax=ax, palette="Set2")
    ax.set_title("Salary by AI Impact Level")
    st.pyplot(fig, use_container_width=True)
    plt.close()
 
    st.markdown("---")
 
    # t-test ──────────────────────────────────────────────────────────────────
    st.markdown("### Independent Samples t-Test")
    st.markdown(
        "**H₀:** Mean salary is equal for Increasing and Decreasing job status.  \n"
        "**H₁:** Mean salaries differ between the two groups."
    )
    inc = df[df["Job Status"] == "Increasing"]["Median Salary (USD)"]
    dec = df[df["Job Status"] == "Decreasing"]["Median Salary (USD)"]
    t_stat, p_ttest = ttest_ind(inc, dec)
 
    da, db, dc = st.columns(3)
    da.metric("t-Statistic", f"{t_stat:.4f}")
    db.metric("P-Value", f"{p_ttest:.4f}")
    dc.metric(
        "Result",
        "Reject H₀ ✅" if p_ttest < 0.05 else "Fail to Reject H₀ ❌",
    )
    if p_ttest < 0.05:
        st.success("Significant difference in salary between Increasing and Decreasing job status (p < 0.05).")
    else:
        st.info("No significant difference in salary between the two job status groups (p ≥ 0.05).")
 
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=df, x="Job Status", y="Median Salary (USD)", ax=ax, palette="Set1")
    ax.set_title("Salary by Job Status")
    st.pyplot(fig, use_container_width=True)
    plt.close()
 
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Feature Selection – SelectKBest (f_regression)")
    st.markdown("Scores show how strongly each numeric feature correlates with the target `Projected Openings (2030)`.")
 
    X_fs = df[NUMERIC_FEATURES]
    y_fs = df[TARGET]
    selector = SelectKBest(score_func=f_regression, k="all")
    selector.fit(X_fs, y_fs)
 
    fs_df = pd.DataFrame({
        "Feature": NUMERIC_FEATURES,
        "F-Score": selector.scores_,
        "P-Value": selector.pvalues_,
    }).sort_values("F-Score", ascending=False).reset_index(drop=True)
 
    st.dataframe(
        fs_df.style.format({"F-Score": "{:.4f}", "P-Value": "{:.4f}"}),
        use_container_width=True,
    )
 
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(fs_df["Feature"], fs_df["F-Score"], color="#4C72B0")
    ax.set_xlabel("F-Score")
    ax.set_title("Feature Importance (SelectKBest)")
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)
    plt.close()
 
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Model Results")
 
    # Linear Regression ───────────────────────────────────────────────────────
    st.markdown("### Linear Regression (sklearn Pipeline)")
    st.caption("Target: `Projected Openings (2030)` · 80/20 train-test split · random_state=42")
 
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:,.2f}")
    m2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
    m3.metric("R²", f"{metrics['R²']:.4f}")
 
    # Actual vs Predicted scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    axes[0].scatter(y_test, pred, alpha=0.3, s=10, color="#4C72B0")
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()
 
    residuals = y_test - pred
    axes[1].scatter(pred, residuals, alpha=0.3, s=10, color="#DD8452")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")
 
    st.pyplot(fig, use_container_width=True)
    plt.close()
 
    st.markdown("---")
 
    # OLS ────────────────────────────────────────────────────────────────────
    st.markdown("### OLS Regression (statsmodels)")
 
    oa, ob, oc, od = st.columns(4)
    oa.metric("R²", ols_summary["R²"])
    ob.metric("Adj. R²", ols_summary["Adj. R²"])
    oc.metric("F-Statistic", ols_summary["F-Statistic"])
    od.metric("Prob (F-stat)", ols_summary["Prob (F-stat)"])
 
    oe, of_, og = st.columns(3)
    oe.metric("AIC", ols_summary["AIC"])
    of_.metric("BIC", ols_summary["BIC"])
    og.metric("Observations", ols_summary["Observations"])
 
    st.markdown("#### Coefficients Table")
    sig_mask = coef_df["P-Value"] < 0.05
    styled = (
        coef_df.style
        .format({"Coefficient": "{:.4f}", "Std Error": "{:.4f}", "P-Value": "{:.4f}"})
        .apply(
            lambda row: ["background-color: #d4f1d4" if sig_mask[row.name] else "" for _ in row],
            axis=1,
        )
    )
    st.dataframe(styled, use_container_width=True)
    st.caption("🟢 Green rows = statistically significant (p < 0.05)")
 
    # Coefficient bar chart (top 10 by absolute value, excluding const)
    top_coef = (
        coef_df[coef_df["Feature"] != "const"]
        .assign(AbsCoef=lambda x: x["Coefficient"].abs())
        .nlargest(10, "AbsCoef")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in top_coef["Coefficient"]]
    ax.barh(top_coef["Feature"], top_coef["Coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title("Top 10 OLS Coefficients (by absolute value)")
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)
    plt.close()
 