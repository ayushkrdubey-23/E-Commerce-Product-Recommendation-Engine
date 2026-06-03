import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

# Load data
products = pd.read_csv("data/products.csv")
users = pd.read_csv("data/users.csv")
interactions = pd.read_csv("data/interactions.csv")

# ----------------------------
# WEIGHTS
# ----------------------------
event_weight = {
    "view": 1,
    "cart": 3,
    "purchase": 5
}

# ----------------------------
# 1. USER BEHAVIOR SCORE
# ----------------------------
user_category_score = {}

for _, row in interactions.iterrows():
    user = row["user_id"]
    product = row["product_id"]
    event = row["event"]

    category = products[products["product_id"] == product]["category"].values[0]
    score = event_weight.get(event, 0)

    if user not in user_category_score:
        user_category_score[user] = {}

    user_category_score[user][category] = user_category_score[user].get(category, 0) + score


# ----------------------------
# 2. GLOBAL POPULARITY SCORE
# ----------------------------
popularity = {}

for _, row in interactions.iterrows():
    pid = row["product_id"]
    event = row["event"]

    score = event_weight.get(event, 0)
    popularity[pid] = popularity.get(pid, 0) + score


# ----------------------------
# 3. CONTENT SIMILARITY SCORE
# ----------------------------
def content_score(row, target_product):
    target = products[products["product_id"] == target_product].iloc[0]

    score = 0

    if row["category"] == target["category"]:
        score += 5

    if row["brand"] == target["brand"]:
        score += 3

    price_diff = abs(row["price"] - target["price"])
    score += max(0, 5 - (price_diff / 20000))

    return score


# ----------------------------
# 4. HYBRID RECOMMENDATION ENGINE
# ----------------------------
def recommend(user_id, top_n=5, seed_product=None):

    seen = interactions[interactions["user_id"] == user_id]["product_id"].tolist()
    category_pref = user_category_score.get(user_id, {})

    results = []

    for _, row in products.iterrows():

        pid = row["product_id"]

        if pid in seen:
            continue

        # Component 1: user preference
        user_score = category_pref.get(row["category"], 0)

        # Component 2: popularity
        pop_score = popularity.get(pid, 0)

        # Component 3: content similarity (if seed exists)
        content = 0
        if seed_product:
            content = content_score(row, seed_product)

        # FINAL HYBRID SCORE
        final_score = (0.5 * user_score) + (0.3 * pop_score) + (0.2 * content)

        results.append((final_score, pid, row["name"], row["category"]))

    # Sort using score (Heap-style logic alternative)
    results.sort(reverse=True)

    return results[:top_n]


# ----------------------------
# DEMO RUN
# ----------------------------
user = "U1"
seed = "P1"

print("\nHYBRID RECOMMENDATIONS FOR USER:", user)

output = recommend(user_id=user, seed_product=seed)

for item in output:
    print(item)


# Save output
df = pd.DataFrame(output, columns=["score", "product_id", "name", "category"])
df.to_csv("outputs/hybrid_recommendations.csv", index=False)

print("\nOUTPUT SAVED: outputs/hybrid_recommendations.csv")
