# make_feature_importance.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('reports/feature_importances.csv')
df = df.sort_values('xgb_importance', ascending=False).head(10)

plt.figure(figsize=(8, 6))
plt.barh(df['feature'], df['xgb_importance'], color='#2E75B6')
plt.gca().invert_yaxis()
plt.xlabel('Importance (Information Gain)')
plt.title('XGBoost Feature Importance (Top 10)')
plt.tight_layout()
plt.savefig('reports/figures/feature_importance_real.png', dpi=300)
print('Saved')