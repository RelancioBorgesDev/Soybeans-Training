import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import random
import shutil


base_dir = 'dataset/treino/'
train_dir = 'dataset/treino_final/'
val_dir = 'dataset/validacao/'
test_dir = 'dataset/teste/'


def split_data(source_dir, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    all_images = os.listdir(source_dir)
    random.shuffle(all_images)

    total_images = len(all_images)
    train_split = int(train_size * total_images)
    val_split = int(val_size * total_images) + train_split

    train_images = all_images[:train_split]
    val_images = all_images[train_split:val_split]
    test_images = all_images[val_split:]

    # Função para copiar imagens e criar a pasta se não existir
    def copy_images(images, src_dir, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)  # Garante que a pasta existe
        for image in images:
            src_path = os.path.join(src_dir, image)
            dest_path = os.path.join(dest_dir, image)
            shutil.copy(src_path, dest_path)

    # Copiar as imagens para os diretórios correspondentes
    copy_images(train_images, source_dir, train_dir)
    copy_images(val_images, source_dir, val_dir)
    copy_images(test_images, source_dir, test_dir)

# Criar as pastas de treino, validação e teste para cada classe
for folder in ['Caterpillar', 'Diabrotica speciosa', 'Healthy']:
    os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
    os.makedirs(os.path.join(val_dir, folder), exist_ok=True)
    os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

# Separar as imagens
for folder in ['Caterpillar', 'Diabrotica speciosa', 'Healthy']:
    split_data(os.path.join(base_dir, folder),
               os.path.join(train_dir, folder),
               os.path.join(val_dir, folder),
               os.path.join(test_dir, folder))

# Parâmetros do modelo
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

# Pré-processamento das imagens e Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalização
    rotation_range=40,  # Rotação
    width_shift_range=0.2,  # Translação horizontal
    height_shift_range=0.2,  # Translação vertical
    shear_range=0.2,  # Cisalhamento
    zoom_range=0.2,  # Zoom
    horizontal_flip=True,  # Espelhamento horizontal
    fill_mode='nearest'  # Preenchimento de pixels vazios
)

val_datagen = ImageDataGenerator(rescale=1. / 255)  # Apenas normalização

# Carregar os dados de treino e validação
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Construção da CNN
model = models.Sequential([
    tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes: Caterpillar, Diabrotica speciosa, Healthy
])

# Compilação do modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Treinamento da CNN
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# Avaliação no conjunto de teste
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Acurácia no conjunto de teste: {test_acc}')

# Plotar gráfico de acurácia e perda
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia Treinamento')
plt.plot(epochs_range, val_acc, label='Acurácia Validação')
plt.legend(loc='lower right')
plt.title('Acurácia no Treinamento e Validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda Treinamento')
plt.plot(epochs_range, val_loss, label='Perda Validação')
plt.legend(loc='upper right')
plt.title('Perda no Treinamento e Validação')
plt.show()
