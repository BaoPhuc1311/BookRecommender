import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

st.title("📚 Hệ Thống Gợi Ý Sách Dựa Trên SVD")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./data/Books_rating.csv.zip", compression="zip")
    
    # Chuẩn hóa cột
    df = df.rename(columns={"User_id": "UserID", "Id": "BookID", "review/score": "Rating", "title": "Title"})
    df = df[["BookID", "UserID", "Rating", "Title"]].dropna().drop_duplicates()

    # Lọc dữ liệu
    min_reviews, max_reviews = 5, 500
    user_counts = df["UserID"].value_counts()
    df = df[df["UserID"].map(user_counts).between(min_reviews, max_reviews)]

    # Encode UserID và BookID
    user_encoder, book_encoder = LabelEncoder(), LabelEncoder()
    df["UserID"] = user_encoder.fit_transform(df["UserID"])
    df["BookID"] = book_encoder.fit_transform(df["BookID"])

    # Cân bằng dữ liệu bằng downsampling
    probabilities = {1: 0.95, 2: 0.9, 3: 0.45, 4: 0.18, 5: 0.074}
    df["Keep"] = df["Rating"].apply(lambda x: np.random.rand() < probabilities[x])
    df_balanced = df[df["Keep"]].drop(columns=["Keep"])
    
    return df_balanced, user_encoder, book_encoder

df_balanced, user_encoder, book_encoder = load_data()

# Hiển thị thông tin dataset
num_users = df_balanced["UserID"].nunique()
num_books = df_balanced["BookID"].nunique()
num_ratings = df_balanced.shape[0]

st.write(f"👤 Số lượng người dùng: {num_users}")
st.write(f"📖 Số lượng sách: {num_books}")
st.write(f"⭐ Số lượng đánh giá: {num_ratings}")

# Danh sách 10 người dùng ngẫu nhiên
st.subheader("👥 Danh sách 10 người dùng ngẫu nhiên:")
random_users = np.random.choice(df_balanced["UserID"].unique(), 10, replace=False)
random_users_decoded = user_encoder.inverse_transform(random_users)
st.write(random_users_decoded)

# Chuẩn bị dữ liệu cho Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_balanced[["UserID", "BookID", "Rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVD
st.subheader("⚙️ Huấn luyện mô hình SVD")
model = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)
model.fit(trainset)

# Đánh giá mô hình
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
st.write(f"📉 RMSE của mô hình: {rmse:.4f}")

# Hàm gợi ý sách (kèm tên sách và rating dự đoán)
def recommend_books(user_id, model, df, book_encoder, n=5):
    all_books = df["BookID"].unique()
    rated_books = df[df["UserID"] == user_id]["BookID"].values
    books_to_predict = [book for book in all_books if book not in rated_books]
    predictions = [model.predict(user_id, book) for book in books_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Lấy thông tin sách gợi ý
    recommended_books = []
    seen_titles = set()
    
    for pred in predictions:
        book_id = pred.iid
        title = df[df["BookID"] == book_id]["Title"].values[0]
        if title not in seen_titles:
            seen_titles.add(title)
            recommended_books.append({"BookID": book_id, "Title": title, "Predicted_Rating": pred.est})
        if len(recommended_books) >= n:
            break

    return recommended_books

# Gợi ý sách
st.subheader("📖 Gợi ý sách cho người dùng")
user_input = st.text_input("Nhập UserID để nhận gợi ý sách:")

if user_input:
    if user_input in user_encoder.classes_:
        user_id = user_encoder.transform([user_input])[0]
        recommended_books = recommend_books(user_id, model, df_balanced, book_encoder)

        if recommended_books:
            st.write(f"📚 **Top 5 sách gợi ý cho user {user_input}:**")
            for book in recommended_books:
                st.write(f"- **{book['Title']}** (ID: {book['BookID']}) - ⭐ {book['Predicted_Rating']:.2f}")
        else:
            st.warning("⚠️ Không có gợi ý phù hợp cho user này.")
    else:
        st.error("⚠️ UserID không tồn tại trong dữ liệu, vui lòng nhập ID khác!")

st.success("✅ Hệ thống gợi ý sách đã sẵn sàng!")
