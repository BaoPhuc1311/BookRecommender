import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import re

# 1. Đọc dữ liệu
df_books = pd.read_csv("data/Books.csv.zip", compression="zip")
df_ratings = pd.read_csv("data/Ratings.csv.zip", compression="zip")

# 2. Lọc sách phổ biến và đảm bảo khớp với df_books
popular_books = df_ratings["ISBN"].value_counts()
popular_books = popular_books[popular_books >= 20].index
df_books = df_books[df_books["ISBN"].isin(popular_books)].copy()

# 3. Xử lý missing values và chuẩn hóa văn bản
df_books["Book-Author"] = df_books["Book-Author"].fillna("unknown_author")
df_books["Publisher"] = df_books["Publisher"].fillna("unknown_publisher")
df_books["Book-Title"] = df_books["Book-Title"].fillna("unknown_title")

# Hàm chuẩn hóa văn bản
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Loại bỏ ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text).strip()  # Chuẩn hóa khoảng trắng
    return text

# Tạo cột đặc trưng text
df_books["features"] = (
    df_books["Book-Title"].apply(clean_text) + " " +
    df_books["Book-Author"].apply(clean_text) + " " +
    df_books["Publisher"].apply(clean_text)
)

# 4. TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Giới hạn số đặc trưng
tfidf_matrix = vectorizer.fit_transform(df_books["features"])

print("Original TF-IDF shape:", tfidf_matrix.shape)

# 5. Giảm chiều bằng TruncatedSVD
k = min(200, tfidf_matrix.shape[1] - 1)  # Chọn k hợp lý, tránh vượt quá số đặc trưng
svd = TruncatedSVD(n_components=k)
reduced_matrix = svd.fit_transform(tfidf_matrix)

print("Reduced TF-IDF shape (after SVD):", reduced_matrix.shape)

# 6. Tính độ tương đồng giữa các sách
cosine_sim = cosine_similarity(reduced_matrix)

# 7. Tạo chỉ mục tìm kiếm nhanh, xử lý trùng lặp
# Sử dụng ISBN để đảm bảo tính duy nhất, nhưng tra cứu bằng tiêu đề
df_books["Book-Title-Lower"] = df_books["Book-Title"].apply(clean_text)
indices = pd.Series(df_books.index, index=df_books["Book-Title-Lower"]).to_dict()

# 8. Hàm gợi ý sách
def get_recommendations(title, cosine_sim=cosine_sim, df=df_books, indices=indices, top_n=10):
    title_cleaned = clean_text(title)
    if title_cleaned not in indices:
        print(f"❌ Không tìm thấy sách: {title}")
        # Tìm gần đúng bằng chuỗi chứa tiêu đề
        matches = df[df["Book-Title-Lower"].str.contains(title_cleaned, case=False, na=False)]
        if not matches.empty:
            print("📚 Gợi ý: Có thể bạn muốn tìm một trong các sách sau:")
            for _, row in matches.head(5).iterrows():
                print(f"- {row['Book-Title']} ({row['Book-Author']})")
        return pd.DataFrame()

    idx = indices[title_cleaned]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices][["Book-Title", "Book-Author", "Publisher"]].reset_index(drop=True)

# 9. Gợi ý thử
book_name = "Harry Potter and the Chamber of Secrets"
print(f"\n📚 Gợi ý sách tương tự '{book_name}':")
recommendations = get_recommendations(book_name)

if not recommendations.empty:
    for i, row in recommendations.iterrows():
        print(f"{i+1}. {row['Book-Title']} - {row['Book-Author']} ({row['Publisher']})")
