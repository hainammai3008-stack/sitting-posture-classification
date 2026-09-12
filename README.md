
# AI Sitting Posture Monitor - Streamlit

Demo có **2 tính năng chính**:

1. Camera realtime
2. Upload ảnh để phân loại tư thế

## Cấu trúc

```text
posture_streamlit_2features/
├── app.py
├── requirements.txt
├── README.md
└── model/
    └── posture_model.keras
```

## 1. Copy model từ Colab

Sau khi train:

```python
model.save("posture_model.keras")
```

Đặt file vào:

```text
model/posture_model.keras
```

## 2. Kiểm tra CLASS_NAMES

Trên Colab:

```python
print(train_ds.class_names)
```

Copy đúng thứ tự sang `CLASS_NAMES` trong `app.py`.

Bản hiện tại đang để:

```python
CLASS_NAMES = [
    "leaning_backward",
    "leaning_left",
    "leaning_right",
    "upright",
]
```

## 3. Kiểm tra preprocessing

Nếu notebook train MobileNetV2 bằng:

```python
tf.keras.applications.mobilenet_v2.preprocess_input
```

giữ:

```python
PREPROCESS_MODE = "mobilenet"
```

Nếu train bằng:

```python
image / 255.0
```

đổi thành:

```python
PREPROCESS_MODE = "0_1"
```

## 4. Chạy local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sau đó mở:

```text
http://localhost:8501
```

## 5. Tính năng Camera

- Dùng `streamlit-webrtc`.
- AI dự đoán liên tục.
- Làm mượt qua 8 frame.
- Nếu sai tư thế liên tục 10 giây:
  - hiển thị cảnh báo
  - browser phát giọng nói tiếng Việt.
- Sau một lần cảnh báo, đợi 15 giây mới cảnh báo lại.

## 6. Tính năng Upload ảnh

- Chấp nhận JPG/JPEG/PNG.
- Resize về 224x224.
- Chạy cùng model với camera.
- Hiển thị:
  - nhãn dự đoán
  - confidence
  - xác suất của từng class.

## 7. Deploy Streamlit Community Cloud

Push project lên GitHub, sau đó:

- chọn repository
- main file: `app.py`
- deploy

Khi deploy Internet, camera cần HTTPS. Streamlit Community Cloud đã cung cấp HTTPS.
