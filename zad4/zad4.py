import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# load data

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# prepare data augmentation

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

# training models

simple_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
simple_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_simple = simple_cnn.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
print(history_simple.history)
simple_cnn.save_weights('simple_cnn_weights.weights.h5')

simple_cnn_augmented = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
simple_cnn_augmented.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_simple_augmented = simple_cnn_augmented.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=10, validation_data=(x_test, y_test))
print(history_simple_augmented.history)
simple_cnn_augmented.save_weights('simple_cnn_weights_augmented.weights.h5')

advanced_cnn = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    # Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(256, activation='relu'),
    # Dropout(0.5),
    Dense(10, activation='softmax')
])
advanced_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_advanced = advanced_cnn.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
print(history_advanced.history)
advanced_cnn.save_weights('advanced_cnn_weights.weights.h5')

advanced_cnn_augmented = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    # Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(256, activation='relu'),
    # Dropout(0.5),
    Dense(10, activation='softmax')
])
advanced_cnn_augmented.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_advanced_augmented = advanced_cnn_augmented.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=10, validation_data=(x_test, y_test))
print(history_advanced_augmented.history)
advanced_cnn_augmented.save_weights('advanced_cnn_weights_augmented.weights.h5')

# base_model = VGG16(include_top=False, input_shape=(32, 32, 3))
# base_model.trainable = True
# vgg_model = Sequential([
    # base_model,
    # Flatten(),
    # Dense(256, activation='relu'),
    # Dense(10, activation='softmax')
# ])
# vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history_vgg = vgg_model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
# print(history_vgg.history)
# vgg_model.save_weights('vgg_cnn_weights.weights.h5')

# base_model = VGG16(include_top=False, input_shape=(32, 32, 3))
# base_model.trainable = True
# vgg_model_augmented = Sequential([
    # base_model,
    # Flatten(),
    # Dense(256, activation='relu'),
    # Dense(10, activation='softmax')
# ])
# vgg_model_augmented.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history_vgg_augmented = vgg_model_augmented.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=5, validation_data=(x_test, y_test))
# print(history_vgg_augmented.history)
# vgg_model_augmented.save_weights('vgg_cnn_weights_augmented.weights.h5')

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False
transfer_model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_transfer = transfer_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
print(history_transfer.history)
transfer_model.save_weights('transfer_cnn_weights.weights.h5')

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False
transfer_model_augmented = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])
transfer_model_augmented.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_transfer_augmented = transfer_model_augmented.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=10, validation_data=(x_test, y_test))
print(history_transfer_augmented.history)
transfer_model_augmented.save_weights('transfer_cnn_weights_augmented.weights.h5')

# plots

def plot_combined_training_history(histories, labels, title):
    fig = go.Figure()
    
    # Add traces for each history
    for history, label in zip(histories, labels):
        # Train Accuracy
        fig.add_trace(go.Scatter(
            y=history.history['accuracy'],
            mode='lines',
            name=f'{label} - Train Accuracy',
            line=dict(dash='solid')
        ))
        # Validation Accuracy
        fig.add_trace(go.Scatter(
            y=history.history['val_accuracy'],
            mode='lines',
            name=f'{label} - Validation Accuracy',
            line=dict(dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Accuracy'),
        legend=dict(title='Legend'),
        template='plotly_white',
        width=1400,
        height=600
    )
    
    # Show plot
    fig.show()

# Prepare inputs
histories = [history_simple, history_advanced, history_advanced_augmented, history_transfer, history_transfer_augmented]
labels = ["Simple CNN", "Advanced CNN", "Advanced CNN, augmented", "VGG16 Transfer", "VGG16 Transfer, augmented"]

# Plot combined history
plot_combined_training_history(histories, labels, "Porównanie dokładności modeli")

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Macierz konfuzji
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Przewidywane')
    plt.ylabel('Prawdziwe')
    plt.title('Macierz konfuzji')
    plt.show()
    
    # Raport klasyfikacji
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

print("Simple cnn:")
evaluate_model(simple_cnn, x_test, y_test)
print("Simple cnn, augmented:")
evaluate_model(simple_cnn_augmented, x_test, y_test)
print("\nAdvanced cnn:")
evaluate_model(advanced_cnn, x_test, y_test)
print("\nAdvanced cnn, augmented:")
evaluate_model(advanced_cnn_augmented, x_test, y_test)
# print("\nVGG:")
# evaluate_model(vgg_model, x_test, y_test)
# print("\nVGG, augmented:")
# evaluate_model(vgg_model_augmented, x_test, y_test)
print("\ntransfer VGG:")
evaluate_model(transfer_model, x_test, y_test)
print("\ntransfer VGG, augmented:")
evaluate_model(transfer_model_augmented, x_test, y_test)

