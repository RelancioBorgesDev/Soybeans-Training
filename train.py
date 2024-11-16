import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks

# Diretórios do dataset
base_dir = 'dataset/treino/'
train_dir = 'dataset/treino_final/'
val_dir = 'dataset/validacao/'
test_dir = 'dataset/teste/'


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


# Função para plotar o histórico de treinamento
def plot_history(history):
    # Verificar as chaves de histórico e usar as disponíveis
    acc = history.history.get('accuracy', history.history.get('acc', []))
    val_acc = history.history.get('val_accuracy', history.history.get('val_acc', []))
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

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



# Função para exibir resultados de avaliação detalhados no terminal
def print_evaluation_results(test_loss, test_acc, class_names, predictions, test_generator):
    print(f"\n{'-' * 40}\nResultados da Avaliação do Modelo")
    print(f"Perda no conjunto de teste: {test_loss:.4f}")
    print(f"Acurácia no conjunto de teste: {test_acc:.4f}")
    print(f"\n{'-' * 40}\nPredições por imagem:")

    for i, filename in enumerate(test_generator.filenames):
        predicted_class = class_names[np.argmax(predictions[i])]
        print(f"Imagem: {filename} - Classe prevista: {predicted_class}")


# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr]
)

# Plotar o histórico de treinamento
plot_history(history)

# Avaliação no conjunto de teste
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Avaliação do modelo no conjunto de teste
test_loss, test_acc = model.evaluate(test_generator)
predictions = model.predict(test_generator)
classes = list(train_generator.class_indices.keys())

# Exibir resultados detalhados no terminal
print_evaluation_results(test_loss, test_acc, classes, predictions, test_generator)
