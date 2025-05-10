# 📁 discord_bot_multiserver
# Bot Discord AI (ML nhẹ) hỗ trợ mỗi server một model riêng, tự tạo file dữ liệu khi thêm server mới

import discord
import os
import pickle
import json
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import os
TOKEN = os.getenv("DISCORD_TOKEN")
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.dm_messages = True  # Bắt tin nhắn DM
client = discord.Client(intents=intents)

# ======= Mẫu intents mặc định =======
def create_default_intents(path):
    default_data = {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["hi", "hello", "xin chào"],
                "responses": ["Chào bạn!", "Hello!"]
            },
            {
                "tag": "goodbye",
                "patterns": ["bye", "tạm biệt"],
                "responses": ["Tạm biệt nhé!", "Hẹn gặp lại!"]
            }
        ]
    }
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/intents.json", "w", encoding="utf-8") as f:
        json.dump(default_data, f, ensure_ascii=False, indent=2)

# ======= Ghi log câu hỏi chưa hiểu vào pending_data.json =======
def save_pending_question(server_id, question, user):
    path = f"data/{server_id}/pending_data.json"
    os.makedirs(f"data/{server_id}", exist_ok=True)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                pending = json.load(f)
        else:
            pending = []
        if any(p["text"].strip().lower() == question.strip().lower() for p in pending):
            print("⚠️ Đã tồn tại câu hỏi tương tự, không lưu lại.")
            return
        pending.append({
            "text": question,
            "from": str(user),
            "tag": None,
            "timestamp": datetime.utcnow().isoformat()
        })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pending, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("❌ Lỗi khi lưu pending_data:", e)

# ======= Xem danh sách câu hỏi chờ xử lý =======
def get_pending_questions(server_id):
    path = f"data/{server_id}/pending_data.json"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ======= Load mô hình theo server =======
def load_model(server_id):
    path = f"data/{server_id}"
    try:
        with open(f"{path}/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"{path}/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open(f"{path}/intents.json", encoding="utf-8") as f:
            intents = json.load(f)
        return model, vectorizer, intents
    except:
        return None, None, None

# ======= Xử lý response =======
def get_response(msg, model, vectorizer, intents, threshold=0.7):
    if not model or not vectorizer:
        return None
    vec = vectorizer.transform([msg.lower()])
    probs = model.predict_proba(vec)[0]
    confidence = max(probs)
    if confidence < threshold:
        return None
    tag = model.predict(vec)[0]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return intent["responses"][0]
    return None

# ======= Lệnh TRAIN đơn giản =======
def train_model(server_id):
    path = f"data/{server_id}"
    os.makedirs(path, exist_ok=True)
    try:
        with open(f"{path}/intents.json", encoding="utf-8") as f:
            data = json.load(f)

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
        return True
    except Exception as e:
        print("Train Error:", e)
        return False

# ======= Sự kiện: Bot được thêm vào server mới =======
@client.event
async def on_guild_join(guild):
    server_id = str(guild.id)
    path = f"data/{server_id}"
    if not os.path.exists(f"{path}/intents.json"):
        create_default_intents(path)
        print(f"📁 Tạo dữ liệu mặc định cho server {server_id}")

# ======= Sự kiện chính =======
@client.event
async def on_ready():
    print(f"🤖 Bot đã sẵn sàng: {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.guild is None:
        await message.channel.send("❌ Bot hiện chưa hỗ trợ tin nhắn trực tiếp (DM). Vui lòng dùng trong server.")
        return

    server_id = str(message.guild.id)

    if message.content.startswith("/train") or message.content.startswith("/refresh"):
        ok = train_model(server_id)
        if ok:
            await message.channel.send("✅ Đã huấn luyện lại mô hình cho server này!")
        else:
            await message.channel.send("❌ Lỗi khi huấn luyện. Kiểm tra file intents.json")
        return

    if message.content.startswith("/pending"):
        pending = get_pending_questions(server_id)
        if not pending:
            await message.channel.send("✅ Không có câu hỏi nào chờ xử lý!")
        else:
            msg = "📋 **Danh sách câu hỏi chưa xử lý:**\n"
            for idx, item in enumerate(pending, start=1):
                msg += f"{idx}. `{item['text']}` - từ {item['from']} ({item.get('timestamp', 'N/A')})\n"
            await message.channel.send(msg)
        return
    if message.content.startswith("/delete"):
        try:
            parts = message.content.split(" ")
            index = int(parts[1]) - 1
            path = f"data/{server_id}/pending_data.json"
            with open(path, "r", encoding="utf-8") as f:
                pending = json.load(f)

            if index < 0 or index >= len(pending):
                await message.channel.send("❌ Số thứ tự không hợp lệ.")
                return

            deleted_question = pending.pop(index)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(pending, f, ensure_ascii=False, indent=2)

            await message.channel.send(f"🗑️ Đã xoá câu hỏi: `{deleted_question['text']}`")
        except Exception as e:
            await message.channel.send(f"❌ Lỗi khi xoá: {e}")
        return

    if message.content.startswith("/reply"):
        try:
            parts = message.content.split(" ", 3)
            index = int(parts[1]) - 1
            tag = parts[2]
            new_response = parts[3]

            # Load dữ liệu hiện tại
            pending = get_pending_questions(server_id)
            intents_path = f"data/{server_id}/intents.json"
            with open(intents_path, "r", encoding="utf-8") as f:
                intents_data = json.load(f)

            # Thêm response mới vào intent tương ứng hoặc tạo intent mới
            found = False
            for intent in intents_data["intents"]:
                if intent["tag"] == tag:
                    intent["patterns"].append(pending[index]["text"])
                    if new_response not in intent["responses"]:
                        intent["responses"].append(new_response)
                    found = True
                    break

            if not found:
                intents_data["intents"].append({
                    "tag": tag,
                    "patterns": [pending[index]["text"]],
                    "responses": [new_response]
                })

            # Ghi lại intents mới
            with open(intents_path, "w", encoding="utf-8") as f:
                json.dump(intents_data, f, ensure_ascii=False, indent=2)

            # Xóa câu hỏi đã xử lý
            del pending[index]
            with open(f"data/{server_id}/pending_data.json", "w", encoding="utf-8") as f:
                json.dump(pending, f, ensure_ascii=False, indent=2)

            await message.channel.send(f"✅ Đã thêm vào intent `{tag}` và xóa khỏi pending.")
        except Exception as e:
            await message.channel.send(f"❌ Lỗi xử lý reply: {str(e)}")
        return
    
    model, vectorizer, intents_data = load_model(server_id)
    clean_msg = message.content.strip().lower()
    response = get_response(clean_msg, model, vectorizer, intents_data)
    if response:
        await message.channel.send(response)
    else:
        await message.channel.send("❓ Mình chưa hiểu ý bạn. Mình đã lưu lại để admin duyệt sau.")
        save_pending_question(server_id, clean_msg, message.author)


client.run(TOKEN)
