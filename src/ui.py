import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data():
    # Load datasets
    df_rating = pd.read_csv("data/Ratings.csv.zip", compression="zip")
    df_user = pd.read_csv("data/Users.csv.zip", compression="zip")
    df_books = pd.read_csv("data/Books.csv.zip", compression="zip")

    # Filter active users and popular books
    active_users = df_rating["User-ID"].value_counts()
    active_users = active_users[active_users >= 20].index
    df_filtered = df_rating[df_rating["User-ID"].isin(active_users)]

    popular_books = df_filtered["ISBN"].value_counts()
    popular_books = popular_books[popular_books >= 20].index
    df_filtered = df_filtered[df_filtered["ISBN"].isin(popular_books)]

    # Merge with user and book data
    df_filtered = df_filtered.merge(df_user[["User-ID", "Location"]], on="User-ID", how="left")
    df_filtered = df_filtered.merge(df_books[["ISBN", "Book-Title"]], on="ISBN", how="left")

    # Encode UserID and BookID
    user_mapping = {id: idx for idx, id in enumerate(df_filtered["User-ID"].unique())}
    book_mapping = {isbn: idx for idx, isbn in enumerate(df_filtered["ISBN"].unique())}
    isbn_to_title = dict(df_filtered[["ISBN", "Book-Title"]].drop_duplicates().values)

    df_filtered["UserID"] = df_filtered["User-ID"].map(user_mapping)
    df_filtered["BookID"] = df_filtered["ISBN"].map(book_mapping)
    df_filtered["Rating"] = df_filtered["Book-Rating"]

    # Filter by location (USA)
    context_location = "usa"
    df_context = df_filtered[df_filtered["Location"].str.contains(context_location, case=False)]

    return df_context, user_mapping, book_mapping, isbn_to_title

# Compute recommendations
def get_recommendations(user_id, df_context, user_mapping, book_mapping, isbn_to_title):
    try:
        # Map user_id to UserID
        if user_id not in user_mapping:
            return []
        user_index = user_mapping[user_id]

        # Create user-item matrix
        user_item_matrix = df_context.pivot(index="UserID", columns="BookID", values="Rating").fillna(0)

        # Apply SVD
        m, n = user_item_matrix.shape
        k = int(np.sqrt(min(m, n)))
        svd = TruncatedSVD(n_components=k)
        U = svd.fit_transform(user_item_matrix)
        Sigma = svd.singular_values_
        VT = svd.components_

        # Get user vector
        r = user_item_matrix.iloc[user_index].values.reshape(1, -1)
        r_reduced = r[:, :VT.shape[1]]
        Unew = np.dot(r_reduced, VT.T) / Sigma

        # Find similar users
        similarities = cosine_similarity(Unew, U).flatten()
        top_10_users = [user for user in similarities.argsort()[::-1] if user != user_index][:10]

        # Generate recommendations
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
                    isbn = [k for k, v in book_mapping.items() if v == book_id][0]
                    title = isbn_to_title.get(isbn, "Unknown Title")
                    recommended_books.append((title, row["Rating"]))
                    seen_books.add(book_id)

                if len(recommended_books) >= 10:
                    break
            if len(recommended_books) >= 10:
                break

        recommended_books.sort(key=lambda x: x[1], reverse=True)
        return recommended_books[:10]
    except Exception as e:
        print(f"Error: {e}")
        return []

# Load data at startup
df_context, user_mapping, book_mapping, isbn_to_title = load_and_preprocess_data()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        user_id = int(request.form.get("user_id"))
        recommendations = get_recommendations(user_id, df_context, user_mapping, book_mapping, isbn_to_title)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
