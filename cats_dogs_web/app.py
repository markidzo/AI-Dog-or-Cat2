# ================= Імпорти =================
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ================= Створюємо Flask додаток =================
app = Flask(__name__)

# Папка для завантажених фото
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # створюємо папку, якщо її немає

# Розмір фото для MobileNetV2
IMG_SIZE = 224

# ================= Завантажуємо натреновану модель =================
# Тут використовується файл, який ми зберегли раніше
model = load_model('cats_dogs_transfer_model.h5')

# ================= Функція для передбачення =================
def predict_single(img_path):
    """
    Завантажує фото, обробляє його та повертає:
    - prediction: ймовірність, що це собака
    - label: текстовий підсумок ("Собака" або "Кіт")
    """
    # Завантажуємо і змінюємо розмір
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)/255.0  # нормалізація пікселів 0-1
    img_array = np.expand_dims(img_array, axis=0)  # додаємо batch dimension

    # Робимо передбачення
    prediction = model.predict(img_array)[0][0]

    # Підсумок
    label = "Собака" if prediction >= 0.5 else "Кіт"
    return prediction, label

# ================= Головна сторінка =================
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Відображає HTML-сторінку.
    Якщо надійшов POST (фото завантажено) — обробляє фото.
    """
    result = None
    label = None
    filename = None

    if request.method == 'POST':
        # Перевіряємо, чи файл надійшов
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)  # зберігаємо файл на сервері
            prediction, label = predict_single(file_path)  # передбачення
            result = f"Ймовірність собаки: {prediction:.2f}, ймовірність кота: {1 - prediction:.2f}"

    # Відправляємо змінні в HTML шаблон
    return render_template('index.html', result=result, label=label, filename=filename)

# ================= Маршрут для показу фото =================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Дозволяє браузеру отримати доступ до завантаженого фото"""
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= Запуск сервера =================
if __name__ == '__main__':
    app.run(debug=True)
