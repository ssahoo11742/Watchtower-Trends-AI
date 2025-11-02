import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
df = pd.read_csv("topic_companies_multitimeframe.csv")
sns.scatterplot(data=df, x="Relevance_Score", y="Change_1W",
                hue="Topic_Keywords", alpha=0.7)
plt.axvline(0.30, ls="--", c="red", label="rel-cut")
plt.show()