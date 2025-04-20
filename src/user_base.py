import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# 1. Đọc dữ liệu
df_rating = pd.read_csv("data/Ratings.csv.zip", compression="zip")
df_user = pd.read_csv("data/Users.csv.zip", compression="zip")

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

# 5. Lọc theo ngữ cảnh Location
context_location = "usa"
df_context = df_filtered[df_filtered["Location"].str.contains(context_location, case=False)]

# 6. Tạo ma trận người dùng - sách
user_item_matrix = df_context.pivot(index="UserID", columns="BookID", values="Rating").fillna(0)

# 7. Trích xuất đặc trưng bằng SVD
m, n = user_item_matrix.shape
k = int(np.sqrt(min(m, n)))

svd = TruncatedSVD(n_components=k)
U = svd.fit_transform(user_item_matrix)
Sigma = svd.singular_values_
VT = svd.components_

# 8. Chọn người dùng test
user_index = 10  # index trong ma trận, không phải User-ID gốc
r = user_item_matrix.iloc[user_index].values.reshape(1, -1)
r_reduced = r[:, :VT.shape[1]]
Unew = np.dot(r_reduced, VT.T) / Sigma

# 9. Tính độ tương đồng và chọn người dùng tương tự
similarities = cosine_similarity(Unew, U).flatten()
top_10_users = [user for user in similarities.argsort()[::-1] if user != user_index][:10]

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

print("📚 Top 10 sách được gợi ý cho người dùng:")
for i, (book_id, rating) in enumerate(recommended_books, 1):
    print(f"{i}. BookID: {book_id} - Rating: {rating:.2f}")
