
# Posture Streamlit Demo — bản sửa theo notebook Colab

Bản này đã chỉnh phần inference để khớp chính xác với notebook
`TMA2_HND_Sitting_Posture_Classification(1).ipynb`.

## Điểm sửa quan trọng

Notebook xây model:

```python
inputs = keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
...
model = keras.Model(inputs, outputs)
```

Do đó `preprocess_input()` đã nằm **bên trong file model `.keras`**.

Notebook khi predict chỉ làm:

```python
img = tf.keras.utils.load_img(image_filename, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
batch = tf.expand_dims(img_array, axis=0)
pred = model.predict(batch)
```

Vì vậy app này:

- KHÔNG gọi `mobilenet_v2.preprocess_input()` bên ngoài model.
- KHÔNG chia pixel cho 255.
- Upload ảnh dùng RGB trực tiếp.
- Camera OpenCV chỉ đổi `BGR -> RGB` một lần.
- Resize đúng `224 x 224`.

## Model

Copy model Colab:

```text
model/image_classifier_v1.keras
```

## Chạy

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Kiểm tra class order

Notebook lấy:

```python
class_names = full_ds.class_names
```

Nếu cần xác nhận, in trên Colab:

```python
print(class_names)
```

và sửa `CLASS_NAMES` trong `app.py` đúng thứ tự.

## Test đối chiếu rất nên làm

Dùng cùng một file ảnh:

1. predict trong notebook Colab;
2. upload chính ảnh đó lên Streamlit.

Class và confidence phải gần như giống nhau.

Nếu upload ảnh khớp Colab nhưng camera thật vẫn kém, nguyên nhân chính nhiều khả năng là domain shift giữa dữ liệu Roboflow và webcam thật.
