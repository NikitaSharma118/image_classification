# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.applications import MobileNetV2
# from keras.models import Model
# from keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from keras.optimizers import Adam
# from  keras.callbacks import EarlyStopping, ModelCheckpoint
# import json

# IMAGE_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 20
# DATASET_PATH = '../dataset'

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=15,
#     zoom_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     horizontal_flip=True,
#     brightness_range=(0.8, 1.2)
# )

# train_data = datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# val_data = datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# base_model.trainable = False  # Freeze base

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.3)(x)
# output = Dense(train_data.num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=output)

# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=EPOCHS,
#     callbacks=[early_stop, checkpoint]
# )

# val_loss, val_acc = model.evaluate(val_data)
# print(f"\n✅ Final Validation Accuracy: {val_acc:.4f} | Validation Loss: {val_loss:.4f}")


# # plt.figure(figsize=(10, 5))
# # plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
# # plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue')
# # plt.title('Accuracy over Epochs')
# # plt.xlabel('Epoch')
# # plt.ylabel('Accuracy')
# # plt.legend()
# # plt.grid(True)
# # plt.show()

# # plt.figure(figsize=(10, 5))
# # plt.plot(history.history['loss'], label='Train Loss', color='red')
# # plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
# # plt.title('Loss over Epochs')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.grid(True)
# # plt.show()

# # model.save("model.h5")
# # with open("class_names.json", "w") as f:
# #     json.dump(train_data.class_indices, f)

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# model = load_model("model.h5")

# # Load class indices
# with open("class_names.json", "r") as f:
#     class_names = json.load(f)
#     class_names = {int(v): k for k, v in class_names.items()} 

# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = img_array / 255.0  # same as training rescaling
#     img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
#     return img_array

# def predict_image(img_path):
#     img_array = preprocess_image(img_path)
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     class_label = class_names[predicted_class]
#     confidence = prediction[0][predicted_class]
#     return class_label, confidence

# # Example usage
# image_path = "./new_data/cat.jpg"
# predicted_class, confidence = predict_image(image_path)
# print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")


import os
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image

# === CONFIG ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
DATASET_PATH = '../dataset'

# === DATA LOADING ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important for confusion matrix
)

# === SAVE CLASS MAPPING ===
with open("class_names.json", "w") as f:
    json.dump(train_data.class_indices, f)

# === BASE MODEL ===
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Fine-tune last 30 layers
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# === TRAINING ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# === FINAL METRICS ===
val_loss, val_acc = model.evaluate(val_data)
print(f"\n✅ Final Validation Accuracy: {val_acc:.4f} | Validation Loss: {val_loss:.4f}")

# === PLOTS ===
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# === CONFUSION MATRIX ===
y_true = val_data.classes
y_pred = model.predict(val_data)
y_pred_labels = np.argmax(y_pred, axis=1)

with open("class_names.json", "r") as f:
    class_dict = json.load(f)
    class_names = {int(v): k for k, v in class_dict.items()}

cm = confusion_matrix(y_true, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
disp.plot(xticks_rotation=90)
plt.title("Confusion Matrix")
plt.show()

# === SAVE FINAL MODEL ===
model.save("model.keras" , save_format="keras")
