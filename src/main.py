import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/Books_rating.csv.zip", compression="zip")

df_filter = df[["Id", "User_id", "review/score"]].dropna()
df_filter.rename(columns={"User_id": "UserID", "Id": "BookID", "review/score": "Rating"}, inplace=True)

user_encoder, book_encoder = LabelEncoder(), LabelEncoder()
df_filter["UserID"], df_filter["BookID"] = user_encoder.fit_transform(df_filter["UserID"]), book_encoder.fit_transform(df_filter["BookID"])

probabilities = {1: 0.78, 2: 1.0, 3: 0.60, 4: 0.24, 5: 0.08}
df_filter["keep"] = df_filter["Rating"].apply(lambda x: np.random.rand() < probabilities[x])
df_balanced = df_filter[df_filter["keep"]].drop(columns=["keep"])

rating_counts = df_balanced["Rating"].value_counts(normalize=True) * 100

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=rating_counts.index, y=rating_counts.values, hue=rating_counts.index, palette="deep", legend=False)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

plt.xlabel("User Rating")
plt.ylabel("Distribution (%)")
plt.title("Distribution of User Ratings")
plt.ylim(0, max(rating_counts.values) + 5)
plt.savefig("./images/user_rating_distribution.png", dpi=300, facecolor="white", bbox_inches="tight", transparent=True)
plt.show()
