fontsize = 18

df = tab_diff[[abscisse] + criteria].dropna().copy()

# 10 blocs
df["bin"] = pd.qcut(df[abscisse], q=quantiles, labels=False, duplicates="drop")

# Moyennes par bloc
df_plot = df.groupby("bin").mean().reset_index(drop=True)

n = len(criteria)
fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

if n == 1:
    axes = [axes]

colors = ['blue', 'red', 'orange', 'green']

for ax, crit, color in zip(axes, criteria, colors):
    ax.plot(df_plot[abscisse], df_plot[crit], 'o-', color=color,
             linewidth=2, markersize=12)
    ax.set_xlabel(abscisse, fontsize=fontsize)
    ax.set_ylabel(crit, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
