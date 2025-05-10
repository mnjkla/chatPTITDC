import json
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# ✅ Nhập ID server bạn muốn huấn luyện
server_id = "123456789012345678"  # 🔁 Thay bằng ID thực tế
path = f"data/{server_id}"

# ✅ Đảm bảo thư mục tồn tại
if not os.path.exists(f"{path}/intents.json"):
    print("❌ intents.json chưa tồn tại trong thư mục server này.")
    exit()

try:
    with open(f"{path}/intents.json", encoding="utf-8") as file:
        data = json.load(file)

    X, y = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            X.append(pattern.lower())
            y.append(intent["tag"])

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    with open(f"{path}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{path}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"✅ Training complete for server {server_id}")

except Exception as e:
    print("❌ Error during training:", e)
