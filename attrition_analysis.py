"""
PROBLEM 03 | EMPLOYEE ATTRITION - Logistic Regression Project
Predict whether an employee is likely to leave within six months.
Dataset: dataset_03_employee_attrition.csv (real, provided)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, RocCurveDisplay)

# ------------------------------------------------------------------
# 1. LOAD & INSPECT
# ------------------------------------------------------------------
df = pd.read_csv('dataset_03_employee_attrition.csv')
print("="*70)
print("1. DATASET SHAPE:", df.shape)
print("="*70)
print(df.dtypes)
print("\nFirst rows:\n", df.head())
print("\nSummary stats:\n", df.describe())

# ------------------------------------------------------------------
# 2. DATA QUALITY CHECKS
# ------------------------------------------------------------------
print("\n" + "="*70)
print("2. DATA QUALITY CHECKS")
print("="*70)
print("Missing values per column:\n", df.isnull().sum())
n_dupes = df.duplicated().sum()
print(f"\nDuplicate rows: {n_dupes}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after dropping duplicates: {df.shape}")

print("\nClass balance (target):")
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True).round(3))

# ------------------------------------------------------------------
# 3. FEATURES / TARGET
# ------------------------------------------------------------------
target = 'target'
feature_cols = ['age', 'monthly_income', 'years_at_company',
                 'job_satisfaction', 'overtime_hours', 'promotion_years']
X = df[feature_cols]
y = df[target]

# ------------------------------------------------------------------
# 4. TRAIN/TEST SPLIT (stratified to preserve class balance -> avoids leakage)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ------------------------------------------------------------------
# 5. PREPROCESSING - scale numeric features (fit on train only, applied to test)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 6. MODEL TRAINING
# ------------------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
print("\nModel configuration:")
print(model)

# ------------------------------------------------------------------
# 7. EVALUATION
# ------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "="*70)
print("7. EVALUATION METRICS")
print("="*70)
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")
print(f"ROC-AUC  : {auc:.3f}")
print("\nFull classification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed','Left'], yticklabels=['Stayed','Left'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Employee Attrition')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

plt.figure(figsize=(5,4))
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title('ROC Curve - Employee Attrition')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()

# ------------------------------------------------------------------
# 8. COEFFICIENT INTERPRETATION
# ------------------------------------------------------------------
coefs = model.coef_[0]
coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': coefs})
coef_df['Odds_Ratio'] = np.exp(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Coefficient', ascending=False)

print("\n" + "="*70)
print("8. FEATURE COEFFICIENTS (sorted, standardized features)")
print("="*70)
print(coef_df.to_string(index=False))
coef_df.to_csv('coefficients.csv', index=False)

plt.figure(figsize=(7,5))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', hue='Feature', legend=False, palette='coolwarm')
plt.title('Logistic Regression Coefficients - Employee Attrition')
plt.tight_layout()
plt.savefig('coefficients_plot.png', dpi=150)
plt.show()

print("\nDone. Files saved: confusion_matrix.png, roc_curve.png, coefficients.csv, coefficients_plot.png")

# ------------------------------------------------------------------
# 9. WRITTEN INTERPRETATION (required discussion, not just numbers)
# ------------------------------------------------------------------
tn, fp, fn, tp = cm.ravel()
print("\n" + "="*70)
print("9. CONFUSION MATRIX - INTERPRETATION")
print("="*70)
print(f"""
Out of {tn+fp+fn+tp} test employees:
  - {tn} who stayed were correctly predicted to stay (True Negatives)
  - {fp} who stayed were incorrectly flagged as at-risk (False Positives)
  - {fn} who left were missed by the model (False Negatives)
  - {tp} who left were correctly caught (True Positives)

Because the target is perfectly balanced (50/50), accuracy is a fair
headline metric here. Precision ({prec:.3f}) and recall ({rec:.3f}) are close
to each other and to accuracy -> the model is not biased toward either class.
ROC-AUC ({auc:.3f}) confirms decent (not outstanding) separation between
employees who leave and those who stay, across all thresholds.
""")

print("="*70)
print("10. COEFFICIENT INTERPRETATION (practical meaning)")
print("="*70)
print("""
Because features were standardized, coefficient magnitudes are directly
comparable:

  job_satisfaction (+): strongest POSITIVE predictor of leaving.
      -> COUNTERINTUITIVE: normally higher satisfaction should reduce
         attrition. Worth flagging to the reader rather than hiding it -
         may reflect a genuine pattern in this dataset (e.g. satisfied,
         high performers being poached externally), a confounding effect
         not captured by only 6 features, or an artifact of how the
         data was generated. Needs domain-expert review before acting on it.

  promotion_years (+): more years since last promotion -> higher odds of
      leaving. Matches typical HR intuition (stalled career -> attrition).

  monthly_income (+): higher income -> higher odds of leaving in this data.
      Also COUNTERINTUITIVE vs typical HR assumptions - same caveat as
      job_satisfaction above.

  overtime_hours (-): more overtime -> LOWER odds of leaving.
      Could reflect highly engaged/committed employees putting in extra
      hours, rather than overworked employees quitting.

  years_at_company (-): longer tenure -> lower odds of leaving.
      Matches typical HR intuition.

  age (-): older employees -> lower odds of leaving.
      Matches typical HR intuition.
""")

print("="*70)
print("11. LIMITATIONS, GENERALIZATION RISKS & IMPROVEMENTS")
print("="*70)
print("""
  - Two coefficients (job_satisfaction, monthly_income) contradict typical
    HR intuition. This should be investigated with domain expertise before
    the model is trusted operationally - it may indicate omitted variables,
    a genuine but surprising pattern, or noise in only 6 available features.
  - Only 6 features are available (no department, role, manager, engagement
    survey text) - likely limits how much signal the model can extract,
    consistent with a moderate (not strong) AUC of 0.739.
  - Logistic Regression assumes a LINEAR log-odds relationship between each
    feature and the outcome. True relationships (e.g. a U-shaped tenure
    effect) would not be captured without adding interaction or polynomial
    terms - especially worth testing given the counterintuitive coefficients.
  - Tree-based models (Random Forest, Gradient Boosting) could be tried as
    a comparison since they capture non-linear/interaction effects that
    logistic regression cannot.
  - The classification threshold (default 0.5) should be tuned based on the
    real business cost of missing an at-risk employee vs. the cost of an
    unnecessary retention intervention.
""")

print("="*70)
print("12. CONCLUSION")
print("="*70)
print(f"""
Logistic Regression achieves {acc:.1%} accuracy and an AUC of {auc:.3f} on this
balanced attrition dataset - a moderate improvement over the 50% a random
guess would achieve. The coefficient analysis surfaces both intuitive
relationships (years_at_company and age reducing attrition odds,
promotion_years increasing them) and some that contradict typical HR
assumptions (job_satisfaction and monthly_income both increasing odds of
leaving), which should be investigated further with domain expertise rather
than acted on directly. Overall this is a reasonable, interpretable
baseline, but the counterintuitive signals suggest richer features or a
non-linear model would strengthen both accuracy and practical trust in
the conclusions.
""")
