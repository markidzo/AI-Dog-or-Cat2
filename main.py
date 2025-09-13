# ================= Імпорт бібліотек =================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk  # Для GUI
from tkinter import filedialog  # Для вибору файлів через вікно

# ================= Параметри =================
IMG_SIZE = 224  # MobileNetV2 очікує 224x224
BATCH_SIZE = 16
EPOCHS = 5

# ================= Генератори даних =================
train_datagen = ImageDataGenerator(
    rescale=1./255,        # нормалізація пікселів (0-1)
    rotation_range=40,     # випадкове обертання
    width_shift_range=0.2, # випадковий зсув по ширині
    height_shift_range=0.2,# випадковий зсув по висоті
    shear_range=0.2,       # зсув по куту (shear)
    zoom_range=0.2,        # випадковий зум
    horizontal_flip=True,  # дзеркальне відображення
    fill_mode='nearest'    # заповнення порожніх пікселів
)

test_datagen = ImageDataGenerator(rescale=1./255)  # тестові фото нормалізуємо

# ================= Завантаження даних =================
train = train_datagen.flow_from_directory(
    "data/train",              # папка з тренувальними фото
    target_size=(IMG_SIZE, IMG_SIZE),  # змінюємо розмір фото
    batch_size=BATCH_SIZE,     # кількість фото у пакеті
    class_mode="binary"        # 0=кіт, 1=собака
)

test = test_datagen.flow_from_directory(
    "data/test",               # папка з тестовими фото
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ================= Transfer Learning =================
# Беремо готову модель MobileNetV2 без верхнього шару (include_top=False)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # не тренуємо готові ваги

# Додаємо нові шари для класифікації котів і собак
x = base_model.output
x = GlobalAveragePooling2D()(x)     # усереднюємо ознаки
x = Dense(128, activation='relu')(x) # прихований шар
x = Dropout(0.5)(x)                 # випадкове вимкнення 50% нейронів
predictions = Dense(1, activation='sigmoid')(x)  # вихід: 0=кіт, 1=собака

# Створюємо модель
model = Model(inputs=base_model.input, outputs=predictions)

# ================= Компіляція моделі =================
model.compile(
    optimizer='adam',              # оптимізатор Adam
    loss='binary_crossentropy',    # функція втрат для 2 класів
    metrics=['accuracy']           # метрика точність
)

# ================= Навчання =================
history = model.fit(
    train,               # тренувальні дані
    validation_data=test, # тестові дані
    epochs=EPOCHS        # кількість епох
)

# ================= Збереження моделі =================
model.save("cats_dogs_transfer_model.h5")
print("Модель збережена у cats_dogs_transfer_model.h5")

# ================= Функція для обробки одного фото =================
def predict_single(img_path, model):
    """Завантажує фото, робить передбачення і виводить ймовірність"""
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))  # завантаження і зміна розміру
    img_array = image.img_to_array(img)/255.0   # нормалізація
    img_array = np.expand_dims(img_array, axis=0)  # додаємо batch
    prediction = model.predict(img_array)[0][0]    # робимо передбачення

    # Вивід ймовірностей
    print(f"Ймовірність, що це собака: {prediction:.2f}")
    print(f"Ймовірність, що це кіт: {1 - prediction:.2f}")
    return prediction  # повертаємо коефіцієнт

# ================= Tkinter GUI для вибору фото =================
def open_file():
    """Відкриває діалог вибору файлу і передає його у predict_single"""
    file_path = filedialog.askopenfilename()  # вибір файлу через вікно
    if file_path:
        predict_single(file_path, model)      # робимо передбачення для вибраного фото

# Створюємо головне вікно
root = tk.Tk()
root.title("Cats vs Dogs Predictor")  # заголовок вікна

# Кнопка для вибору фото
button = tk.Button(root, text="Вибрати фото", command=open_file)
button.pack(pady=20)  # додаємо кнопку у вікно з відступом

# Запускаємо GUI
root.mainloop()
