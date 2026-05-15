# 🚀 Streamlit Cloud Deployment Checklist

## ✅ Các bước đã hoàn thành

### 1. **Dependencies Management** ✓
   - ✅ `requirements.txt` tạo sẵn với tất cả thư viện cần thiết
   - Các package chính:
     - streamlit==1.32.2
     - torch==2.0.1
     - pyedflib==0.1.34
     - numpy, scipy, matplotlib, pandas

### 2. **Streamlit Configuration** ✓
   - ✅ `.streamlit/config.toml` cấu hình giao diện dark theme
   - ✅ `.streamlit/secrets.toml` template cho API keys (tùy chọn)
   - ✅ `.gitignore` để tránh upload file không cần thiết

### 3. **Code Fixes for Cloud** ✓
   - ✅ **Fix path issues**: 
     - Sử dụng `Path(__file__).resolve().parent` thay vì hardcoded paths
     - Default checkpoint dir: `./weight/` (nơi models được lưu)
   - ✅ **Fix temp directory**:
     - Thay `os.makedirs("_tmp_inf")` bằng `tempfile.TemporaryDirectory()`
     - Đảm bảo hoạt động trên Streamlit Cloud (read-only filesystem)
   - ✅ **Better error handling**:
     - Thêm kiểm tra file tồn tại
     - PyTorch compatibility (weights_only parameter)

### 4. **Documentation** ✓
   - ✅ `DEPLOYMENT.md` - Hướng dẫn chi tiết deploy
   - ✅ `test_pipeline.py` - Script kiểm tra trước deployment

## 📋 Để Deploy, hãy làm theo:

### **Bước 1: Kiểm tra Local (Optional)**
```bash
cd /workspaces/SleepEEG/PipelineEEG
python test_pipeline.py
```

### **Bước 2: Commit & Push lên GitHub**
```bash
cd /workspaces/SleepEEG
git add -A
git commit -m "Deploy: Fix Streamlit Cloud compatibility"
git push origin main
```

### **Bước 3: Deploy trên Streamlit Cloud**

1. Truy cập: https://share.streamlit.io/
2. Đăng nhập bằng GitHub account của bạn
3. Click **"New app"**
4. Điền thông tin:
   - **Repository**: `quanuiter/SleepEEG`
   - **Branch**: `main`
   - **Main file path**: `PipelineEEG/app.py`
5. Click **"Deploy"** ✨

Streamlit sẽ tự động:
- ✅ Clone repository
- ✅ Cài đặt dependencies từ `requirements.txt`
- ✅ Chạy `app.py`

## 🎯 Vấn đề Đã Giải Quyết

### ❌ Problem 1: "No module found" / Import errors
**Solution:** 
- Thêm `sys.path.insert(0, str(APP_DIR))`
- Tất cả imports sử dụng relative paths

### ❌ Problem 2: "Model weights not found"
**Solution:**
- Default path: `./weight/` (tương đối với app.py)
- App tự động detect ResNet + TCN files
- User có thể chỉ định path khác nếu cần

### ❌ Problem 3: "Permission denied creating temp directory"
**Solution:**
- Thay `os.makedirs("_tmp_inf")` → `tempfile.TemporaryDirectory()`
- Tự động cleanup sau xử lý

### ❌ Problem 4: PyTorch version compatibility
**Solution:**
- Try/except cho `weights_only=True` parameter
- Fallback để compatible với PyTorch cũ hơn

## 📊 App Structure Sau Deploy

```
streamlit app.py (running on cloud)
  ├── Upload EDF file
  ├── Auto-detect channels
  ├── Configure filters
  ├── Select ResNet weights (./weight/resnet_foldN.pth)
  ├── Select TCN weights (./weight/tcn_foldN.pth)
  ├── Run Inference
  │   ├── Preprocess
  │   ├── Feature extraction (ResNet)
  │   ├── Sequence classification (TCN)
  │   └── Generate Hypnogram
  └── Display Results
      ├── Summary stats
      ├── Raw EEG visualization
      ├── Hypnogram chart
      └── Stage distribution
```

## ⚠️ Important Notes

1. **Model Weights Size**: 
   - Tổng ~200MB (ResNet + TCN)
   - Phải có sẵn trong `weight/` folder
   - Git sẽ lưu trữ (nếu dưới 100MB/file)

2. **Inference Speed**:
   - CPU: 1-5 phút tùy độ dài EDF
   - GPU (nếu có): 30-60 giây
   - Streamlit Cloud có thể timeout nếu > 12 phút

3. **File Upload Limit**:
   - Streamlit Cloud: 500MB max (cấu hình sẵn)
   - EDF files thường 10-100MB OK

4. **Data Privacy**:
   - Uploaded files không được lưu trữ
   - Results không persist giữa sessions
   - Tất cả processing trên Streamlit Cloud servers

## 🔗 Resources

- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Cloud: https://share.streamlit.io/
- Deployment Guide: Xem `DEPLOYMENT.md`

---

**Status**: ✅ Ready for Streamlit Cloud Deployment
**Date**: 2026-05-15
**Version**: 2.0 - Cloud Compatible
