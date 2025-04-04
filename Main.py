from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#  Load dataset function
def load_images(folder_path, img_size=(128, 128)):
    images, labels = [], []
    
    for category in ["Real", "Fake"]:
        category_path = os.path.join(folder_path, category)
        
        for filename in os.listdir(category_path):
            image_path = os.path.join(category_path, filename)
            img = cv2.imread(image_path)
            
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(category)

    X = np.array(images, dtype="float32") / 255.0  # Normalize
    y = np.array([1 if label == "Fake" else 0 for label in labels])   # Encode labels (0 = Real, 1 = Fake)
    return X, y

#  Load dataset
X, y = load_images("Frame_Dataset", img_size=(128, 128))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

#  Use Xception as Feature Extractor
base_model = Xception(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze Xception layers

#  Custom Fully Connected Layers on Top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Convert features to a smaller representation
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)  # Prevent overfitting
output = Dense(1, activation="sigmoid")(x)  # Binary classification (Real vs. Fake)

#  Define Final Model
model = Model(inputs=base_model.input, outputs=output)

#  Compile the Model
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

#  Callbacks for Early Stopping and Best Model Saving
checkpoint = ModelCheckpoint("best_xception_model.keras", monitor="val_accuracy", save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

#  Train Model
batch_size = 32
train_generator = datagen.flow(
    X_train, 
    y_train,
    batch_size=batch_size,
    shuffle=True
)

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) // batch_size,
    callbacks=[checkpoint, early_stop]
)

#  Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

#  Save the Final Model
model.save("my_xception_model.keras")
