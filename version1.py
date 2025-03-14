import tensorflow as tf
import numpy as np
import hashlib
import random
import string
import os
import pandas as pd

# 🔹 Параметры модели
MAX_SEQ_LENGTH = 10
HASH_LENGTH = 64  # SHA-256 хэш в HEX
CHARACTER_SET = string.ascii_letters + string.digits  # "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
EMBEDDING_DIM = 512  # Увеличили размерность
HIDDEN_UNITS = 1024  # Больше нейронов в слоях
LEARNING_RATE = 0.0005
BATCH_SIZE = 32

DATASET_FILE = "dataset.csv"
MODEL_PATH = "sha256_decoder.keras"

# 🔹 Функции генерации данных
def generate_random_text():
    """Генерирует случайную строку длиной от 1 до 10 символов."""
    length = random.randint(1, MAX_SEQ_LENGTH)
    return ''.join(random.choices(CHARACTER_SET, k=length))

def sha256_hash(text):
    """Вычисляет SHA-256 хэш строки и возвращает его в hex-формате."""
    return hashlib.sha256(text.encode()).hexdigest()

def generate_dataset(size):
    """Генерирует датасет из случайных строк и их SHA-256 хэшей."""
    data = [(generate_random_text(), None) for _ in range(size)]
    return [(text, sha256_hash(text)) for text, _ in data]

# 🔹 Работа с файлом датасета
def load_dataset():
    """Загружает датасет из файла, если он существует."""
    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE, header=None)
        return list(zip(df[0], df[1]))
    return []

def save_dataset(dataset):
    """Сохраняет датасет в CSV."""
    df = pd.DataFrame(dataset)
    df.to_csv(DATASET_FILE, index=False, header=False)

# 🔹 Преобразование данных
def text_to_one_hot(text, max_length, char_to_idx):
    """Преобразует текст в one-hot представление."""
    seq = [char_to_idx.get(c, 0) for c in text]
    return seq + [0] * (max_length - len(seq))  # Дополняем нулями до max_length

def hash_to_vector(hash_str):
    """Преобразует SHA-256 хэш в числовой вектор (по ASCII-кодам)."""
    return [ord(c) for c in hash_str]

def prepare_data(dataset, char_to_idx):
    """Готовит входные и выходные данные для обучения."""
    x_data = np.array([hash_to_vector(h) for _, h in dataset])
    y_data = np.array([text_to_one_hot(t, MAX_SEQ_LENGTH, char_to_idx) for t, _ in dataset])
    return x_data, y_data

# 🔹 Создание модели
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(HASH_LENGTH,)),

        # Встраивание векторов
        tf.keras.layers.Embedding(input_dim=256, output_dim=EMBEDDING_DIM),

        # Многослойный LSTM
        tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(HIDDEN_UNITS),
        
        # Полносвязные слои
        tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
        tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu"),

        # Выходной слой
        tf.keras.layers.Dense(MAX_SEQ_LENGTH * len(CHARACTER_SET), activation="softmax"),
        tf.keras.layers.Reshape((MAX_SEQ_LENGTH, len(CHARACTER_SET)))
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# 🔹 Загрузка / сохранение модели
def load_model():
    if os.path.exists(MODEL_PATH):
        print("🔄 Загружаем обученную модель...")
        return tf.keras.models.load_model(MODEL_PATH)
    print("🆕 Создаём новую модель...")
    return create_model()

def save_model(model):
    model.save(MODEL_PATH)
    print("💾 Модель сохранена!")

# 🔹 Основной цикл обучения
if __name__ == "__main__":
    model = load_model()

    char_to_idx = {char: idx for idx, char in enumerate(CHARACTER_SET)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    dataset = load_dataset()
    dataset_size = len(dataset)  # Загружаем существующий размер датасета
    print(f"📂 Найдено {dataset_size} примеров в датасете.")

    for i in range(9999):  # Долгое обучение
        new_samples = 25  # Добавляем 25 новых примеров
        print(f"\n🔹 Итерация {i+1}: добавляем {new_samples} примеров...")
        
        new_data = generate_dataset(new_samples)
        dataset.extend(new_data)
        save_dataset(dataset)  # Сохраняем обновленный датасет

        x_train, y_train = prepare_data(dataset, char_to_idx)

        # Преобразование y_train в one-hot представление
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=len(CHARACTER_SET))

        print(f"🚀 Обучение на {len(dataset)} примерах...")
        model.fit(x_train, y_train_one_hot, epochs=25, batch_size=BATCH_SIZE)

        save_model(model)  # Сохранение модели после каждой итерации

    print("✅ Обучение завершено!")
