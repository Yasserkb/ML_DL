#lab 22 : Classification des legumes et des fruits
# Realise par Yasser Koubachi
# ref datasets https://colab.research.google.com/drive/1ONNuxDzGwPMM3ZM7zJEYnmhukGP6cuS5#scrollTo=8lCi3O2Otp9D&line=1&uniqifier=1
# ref source :  https://colab.research.google.com/drive/1ONNuxDzGwPMM3ZM7zJEYnmhukGP6cuS5#scrollTo=8lCi3O2Otp9D
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy

#Step 1 : Datasets
# Data manupilation
img_height, img_width = 32, 32
batch_size = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    "../Datasets/fruits/train",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "../Datasets/fruits/validation",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "../Datasets/fruits/test",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
#data visualisation::
class_names = ["apple", "banana", "orange"]
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
#Step 2: Model
model = tf.keras.Sequential(
    [
     tf.keras.layers.Rescaling(1./255),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation="relu"),
     tf.keras.layers.Dense(3)
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)
#step 3: Train
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10
)
#Step 4: Test
model.evaluate(test_ds)

plt.figure(figsize=(10,10))
for images, labels in test_ds.take(1):
  classifications = model(images)
  # print(classifications)

  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    index = numpy.argmax(classifications[i])
    plt.title("Pred: " + class_names[index] + " | Real: " + class_names[labels[i]])
plt.show()
# Streamlit for web deployment :
# how to include this model to flutter or web search for model.tflite to a mobile app
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", 'wb') as f:
  f.write(tflite_model)
  