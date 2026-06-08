# Dùng phiên bản Python nhẹ
FROM python:3.11-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Copy file requirements.txt vào trước để cài thư viện
COPY requirements.txt .

# Cài đặt các thư viện (không lưu cache để image nhẹ hơn)
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code và model từ máy thật vào container
COPY . .

# Mở cổng 8501 cho Streamlit
EXPOSE 8501

# Lệnh chạy app khi container khởi động
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]