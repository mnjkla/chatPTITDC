📘 README - Discord Chatbot AI Đa Server

1. Giới thiệu
   Bot Discord này sử dụng mô hình học máy (ML nhẹ) để phản hồi tin nhắn người dùng. Mỗi server Discord sẽ có một bộ dữ liệu riêng (intents.json), mô hình riêng (model.pkl) và bộ vectorizer riêng (vectorizer.pkl). Bot có khả năng học thêm từ phản hồi của admin và hỗ trợ nhiều lệnh slash như /train, /pending, /reply.
2. Tính năng chính

- Phản hồi tự động dựa trên intents mẫu huấn luyện.
- Mỗi server có thể tự huấn luyện và lưu mô hình riêng.
- Ghi lại câu hỏi chưa hiểu và chờ admin phản hồi.
- Admin có thể trả lời và bổ sung vào tập dữ liệu trực tiếp bằng lệnh /reply.
- Hỗ trợ các slash command tiện lợi như /train, /pending, /delete, /reply.

3. Cài đặt
1. Cài Python 3.10+ và pip.
1. Cài thư viện cần thiết:
   pip install -r requirements.txt
1. Đặt token bot vào file .env hoặc trong biến TOKEN trong mã.
1. Chạy bot bằng:
   python bot_multi.py
1. Cấu trúc thư mục

📁 discord_chatbot/
├── bot_multi.py # File chính chạy bot
├── train_model.py # Huấn luyện mô hình từ intents.json
├── requirements.txt # Thư viện cần cài
├── token.txt # Chứa token (nếu dùng cách tách riêng)
├── data/
│ ├── <server_id>/
│ │ ├── intents.json
│ │ ├── model.pkl
│ │ ├── vectorizer.pkl
│ │ └── pending_data.json

5. Slash Commands
   /train - Huấn luyện lại mô hình cho server hiện tại.
   /pending - Xem các câu hỏi chưa hiểu.
   /reply <index> <tag> <response> - Trả lời và lưu câu hỏi vào mô hình.
   /delete <index> - Xóa câu hỏi khỏi danh sách chờ.
6. Ghi chú

- Bot chỉ trả lời khi có confidence trên 40%.
- Nếu không hiểu, bot sẽ lưu lại câu hỏi để admin xử lý.

Chúc bạn sử dụng bot hiệu quả! 🚀
