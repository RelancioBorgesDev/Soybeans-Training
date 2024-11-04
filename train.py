import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import os
import random
import shutil

# Diretórios do dataset
base_dir = 'dataset/treino/'
train_dir = 'dataset/treino_final/'
val_dir = 'dataset/validacao/'
test_dir = 'dataset/teste/'


# Função para dividir o dataset entre treino, validação e teste
def split_data(source_dir, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    all_images = os.listdir(source_dir)
    random.shuffle(all_images)

    total_images = len(all_images)
    train_split = int(train_size * total_images)
    val_split = int(val_size * total_images) + train_split

    train_images = all_images[:train_split]
    val_images = all_images[train_split:val_split]
    test_images = all_images[val_split:]

    def copy_images(images, src_dir, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        for image in images:
            src_path = os.path.join(src_dir, image)
            dest_path = os.path.join(dest_dir, image)
            shutil.copy(src_path, dest_path)

    copy_images(train_images, source_dir, train_dir)
    copy_images(val_images, source_dir, val_dir)
    copy_images(test_images, source_dir, test_dir)


# Dividir o dataset para cada classe
for folder in ['Caterpillar', 'Diabrotica speciosa', 'Healthy']:
    split_data(os.path.join(base_dir, folder),
               os.path.join(train_dir, folder),
               os.path.join(val_dir, folder),
               os.path.join(test_dir, folder))

# Parâmetros de treinamento
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# Normalização apenas, sem aumento de dados adicional
train_datagen = ImageDataGenerator(rescale=1. / 255)

val_datagen = ImageDataGenerator(rescale=1. / 255)

# Geradores de dados
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

# Definição do modelo de CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Compilação do modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr]
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

# Plotar gráficos de acurácia e perda
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
