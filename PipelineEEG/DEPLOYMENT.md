# Sleep EEG Pipeline — Streamlit Cloud Deployment Guide

## 📋 Yêu cầu

- Git repository với toàn bộ code
- Model weights (.pth files) trong thư mục `weight/`
- Python 3.8+

## 🚀 Deploy trên Streamlit Cloud

### 1. **Chuẩn bị GitHub Repository**

```bash
cd PipelineEEG
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

**Đảm bảo file có sẵn:**
- ✅ `requirements.txt` — dependencies
- ✅ `.streamlit/config.toml` — cấu hình giao diện
- ✅ `app.py` — ứng dụng chính
- ✅ `weight/*.pth` — model weights (ResNet + TCN)

### 2. **Deploy trên Streamlit Cloud**

Truy cập [share.streamlit.io](https://share.streamlit.io/) và:

1. Đăng nhập bằng GitHub account
2. Click "New app"
3. Chọn:
   - **Repository**: `quanuiter/SleepEEG`
   - **Branch**: `main`
   - **Main file path**: `PipelineEEG/app.py`
4. Click "Deploy"

### 3. **Kiểm tra Model Weights**

App sẽ tự động tìm kiếm weights trong thư mục `./weight/`:

```
PipelineEEG/
  weight/
    resnet_fold1.pth
    resnet_fold2.pth
    ...
    tcn_fold1.pth
    tcn_fold2.pth
    ...
```

Nếu không tìm thấy, hãy:
- Upload file `.pth` vào thư mục `weight/`
- Hoặc chỉ định đường dẫn custom trong UI

## ⚙️ Cách Dùng

### Tab "INFERENCE" (Dự đoán Sleep Stages)

1. **Upload EDF file**
   - Hỗ trợ: Sleep-EDF hoặc SHHS format
   - Tệp chứa kênh EEG

2. **Chọn Channel & Bộ Lọc**
   - Auto-detect hoặc manual select
   - Tuỳ chỉnh bandpass filter (0.5-30 Hz mặc định)
   - Notch filter tại 50 Hz

3. **Chọn Model Weights**
   - ResNet (feature extraction)
   - TCN (sequence classification)

4. **Chạy Inference**
   - Preprocessor → ResNet → TCN
   - Kết quả: Hypnogram + visualization

### Output

- 📊 **Hypnogram** — Biểu đồ 5 giai đoạn ngủ (W/N1/N2/N3/REM)
- 🌊 **Raw Signal Plots** — Các epoch EEG đã lọc
- 📈 **Statistics** — Thống kê phân bố giai đoạn

## 🔧 Troubleshooting

### ❌ "Model weights not found"
- Kiểm tra file `.pth` có trong thư mục `weight/` không
- Nếu upload EDF không thành công, kiểm tra format và channel names

### ❌ "Channel not found in EDF"
- Click "Xem danh sách channel" để thấy các channel có sẵn
- Hoặc nhập channel name thủ công

### ❌ Streamlit Cloud timeout
- Model inference có thể mất vài phút
- Hãy chờ hoặc tối ưu EDF file (giảm thời gian ghi)

## 📦 Dependencies

Tất cả được quản lý bởi `requirements.txt`:

```
streamlit==1.32.2
numpy==1.24.3
scipy==1.11.4
torch==2.0.1
pyedflib==0.1.34
matplotlib==3.8.2
pandas==2.1.3
scikit-learn==1.3.2
```

## 📝 Ghi chú

- **Streamlit Cloud Storage**: Filesystem là read-only, chỉ có `/tmp` có thể ghi
- **Cached Results**: Kết quả inference không lưu giữa sessions (thiết kế có ý)
- **Model Size**: Tổng ~200MB (ResNet + TCN weights)

---

**Mọi lỗi hoặc câu hỏi?** Hãy kiểm tra Streamlit logs trên dashboard: `App > Manage app > View logs`
