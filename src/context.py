import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

# 1. Đọc dữ liệu
df_rating = pd.read_csv("data/Ratings.csv.zip", compression="zip")
df_user = pd.read_csv("data/Users.csv.zip", compression="zip")

# Simulate a timestamp column for context (since dataset doesn't have it)
np.random.seed(42)
start_date = datetime(2023, 1, 1)
df_rating["Rating-Timestamp"] = [
    (start_date + timedelta(days=np.random.randint(0, 730))).strftime("%Y-%m-%d")
    for _ in range(len(df_rating))
]

# 2. Lọc người dùng hoạt động và sách phổ biến
active_users = df_rating["User-ID"].value_counts()
active_users = active_users[active_users >= 20].index
df_filtered = df_rating[df_rating["User-ID"].isin(active_users)]

popular_books = df_filtered["ISBN"].value_counts()
popular_books = popular_books[popular_books >= 20].index
df_filtered = df_filtered[df_filtered["ISBN"].isin(popular_books)]

# 3. Thêm Location từ bảng user
df_filtered = df_filtered.merge(df_user[["User-ID", "Location"]], on="User-ID", how="left")

# 4. Mã hóa lại UserID và BookID
user_mapping = {id: idx for idx, id in enumerate(df_filtered["User-ID"].unique())}
book_mapping = {isbn: idx for idx, isbn in enumerate(df_filtered["ISBN"].unique())}

df_filtered["UserID"] = df_filtered["User-ID"].map(user_mapping)
df_filtered["BookID"] = df_filtered["ISBN"].map(book_mapping)
df_filtered["Rating"] = df_filtered["Book-Rating"]

# 5. Định nghĩa ngữ cảnh (context)
context_location = "usa"
context_time = "recent"  # Ratings within the last 6 months
cutoff_date = (datetime(2024, 1, 1) - timedelta(days=180)).strftime("%Y-%m-%d")

# Lọc dữ liệu theo ngữ cảnh
df_context = df_filtered[
    (df_filtered["Location"].str.contains(context_location, case=False, na=False)) &
    (df_filtered["Rating-Timestamp"] >= cutoff_date)
]

# 6. Tạo ma trận người dùng - sách theo ngữ cảnh
user_item_matrix = df_context.pivot_table(
    index="UserID", columns="BookID", values="Rating", aggfunc="mean"
).fillna(0)

# 7. Trích xuất đặc trưng bằng SVD
m, n = user_item_matrix.shape
k = int(np.sqrt(min(m, n)))

svd = TruncatedSVD(n_components=k)
U = svd.fit_transform(user_item_matrix)
Sigma = svd.singular_values_
VT = svd.components_

# 8. Chọn người dùng test và ngữ cảnh
user_index = 10  # index trong ma trận
r = user_item_matrix.iloc[user_index].values.reshape(1, -1)
r_reduced = r[:, :VT.shape[1]]
Unew = np.dot(r_reduced, VT.T) / Sigma

# 9. Tính độ tương đồng có trọng số theo ngữ cảnh
# Thêm trọng số dựa trên sự tương đồng của ngữ cảnh (Location và Time)
similarities = cosine_similarity(Unew, U).flatten()

# Điều chỉnh độ tương đồng dựa trên ngữ cảnh
context_weights = np.ones(len(user_item_matrix))
for idx in range(len(user_item_matrix)):
    user_id = user_item_matrix.index[idx]
    user_ratings = df_context[df_context["UserID"] == user_id]
    # Trọng số cao hơn nếu người dùng có nhiều đánh giá trong ngữ cảnh tương tự
    location_match = user_ratings["Location"].str.contains(context_location, case=False, na=False).mean()
    time_match = (user_ratings["Rating-Timestamp"] >= cutoff_date).mean()
    context_weights[idx] = 0.6 * similarities[idx] + 0.2 * location_match + 0.2 * time_match

# Chọn top 10 người dùng tương tự
top_10_users = [user for user in context_weights.argsort()[::-1] if user != user_index][:10]

# 10. Gợi ý sách dựa trên các người dùng tương đồng
test_user_rated_books = set(user_item_matrix.iloc[user_index].to_numpy().nonzero()[0])
recommended_books = []
seen_books = set()

for user in top_10_users:
    user_top_books = df_context[
        (df_context["UserID"] == user) & (df_context["Rating"] > 3)
    ].sort_values(by="Rating", ascending=False)

    for _, row in user_top_books.iterrows():
        book_id = row["BookID"]
        if book_id not in seen_books and book_id not in test_user_rated_books:
            recommended_books.append((book_id, row["Rating"]))
            seen_books.add(book_id)

        if len(recommended_books) >= 10:
            break
    if len(recommended_books) >= 10:
        break

# 11. Hiển thị kết quả
recommended_books.sort(key=lambda x: x[1], reverse=True)

print(f"📚 Top 10 sách được gợi ý cho người dùng trong ngữ cảnh (Location: {context_location}, Time: {context_time}):")
for i, (book_id, rating) in enumerate(recommended_books, 1):
    print(f"{i}. BookID: {book_id} - Rating: {rating:.2f}")
