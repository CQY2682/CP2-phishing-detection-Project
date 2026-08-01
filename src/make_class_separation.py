"""
Builds Figure 4.x (class separation) from YOUR real data, not synthetic.
Run from your project root:  python make_class_separation_figure.py

Needs: data/X_train.csv (has url_length, has_https columns) and data/y_train.csv (label; 1=phishing, 0=legit)
Output: reports/figures/class_separation.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = pd.read_csv('data/X_train.csv')
y = pd.read_csv('data/y_train.csv')['label']

legit_len = X.loc[y == 0, 'url_length']
phish_len = X.loc[y == 1, 'url_length']
https_legit = X.loc[y == 0, 'has_https'].mean() * 100
https_phish = X.loc[y == 1, 'has_https'].mean() * 100

LEG, PHI = '#2563eb', '#dc2626'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# Chart 1: real URL-length histograms, clipped at 160 chars so the tail does not flatten the plot
bins = np.linspace(0, 160, 55)
ax1.hist(legit_len.clip(upper=160), bins=bins, color=LEG, alpha=0.75,
         label=f'Legitimate (mean {legit_len.mean():.1f})')
ax1.hist(phish_len.clip(upper=160), bins=bins, color=PHI, alpha=0.6,
         label=f'Phishing (mean {phish_len.mean():.1f})')
ax1.axvline(legit_len.mean(), color=LEG, lw=1.6, ls='--')
ax1.axvline(phish_len.mean(), color=PHI, lw=1.6, ls='--')
ax1.set_xlabel('URL length (characters, capped at 160 for display)')
ax1.set_ylabel('Number of URLs')
ax1.set_title('URL length: legitimate short and tight, phishing long and spread',
              fontsize=11, fontweight='bold')
ax1.set_xlim(0, 160)
ax1.legend(fontsize=8.5)
ax1.spines[['top', 'right']].set_visible(False)

# Chart 2: real HTTPS adoption
bars = ax2.bar(['Legitimate', 'Phishing'], [https_legit, https_phish],
               color=[LEG, PHI], width=0.55)
for b, v in zip(bars, [https_legit, https_phish]):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.2f}%',
             ha='center', fontweight='bold', fontsize=12)
ax2.set_ylabel('URLs using HTTPS (%)')
ax2.set_ylim(0, 112)
ax2.set_title('HTTPS use: every legitimate URL, only half of phishing',
              fontsize=11, fontweight='bold')
ax2.spines[['top', 'right']].set_visible(False)

fig.suptitle('The two classes are structurally easy to separate in PhiUSIIL',
             fontsize=12.5, fontweight='bold', y=1.02)
plt.tight_layout()
os.makedirs('reports/figures', exist_ok=True)
out = 'reports/figures/class_separation.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print('Saved', out)
print(f'legit mean len {legit_len.mean():.2f} | phish mean len {phish_len.mean():.2f}')
print(f'https legit {https_legit:.2f}% | https phish {https_phish:.2f}%')