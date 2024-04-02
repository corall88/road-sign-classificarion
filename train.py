import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

# Задаем параметры
train_data_dir = r'C:\Users\пк\Desktop\proging\Python\ML\science project\train_dataset'
test_data_dir = r'C:\Users\пк\Desktop\proging\Python\ML\science project\test_dataset'
img_width, img_height = 150, 150
batch_size = 15
epochs = 25

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

inputs = Input(shape=(img_width, img_height, 3))

x = Conv2D(64, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
# Выходной слой
outputs = Dense(1, activation='sigmoid')(x)

# Создаем модель
model = Model(inputs, outputs)

# Компилируем модель
model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

checkpoint_path = 'best_model.keras'

# Создаем callback для сохранения модели с лучшей точностью
checkpoint = ModelCheckpoint(checkpoint_path, 
                             monitor='val_accuracy',  # Мониторим точность на валидационном наборе данных
                             save_best_only=True,  # Сохраняем только лучшую модель
                             mode='max',  # Режим максимизации (т.е. большая точность лучше)
                             verbose=1  # Выводим информацию о сохранении модели
                            )


# Обучаем модель
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,  # Указываем тестовый генератор для оценки на каждой эпохе
    callbacks=[checkpoint]
    )

best_model = load_model(checkpoint_path)

# Оцениваем лучшую модель на тестовом наборе данных
test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=1)
print("Best Test Loss:", test_loss)
print("Best Test Accuracy:", test_accuracy)