import os
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Parameters
input_dir = 'D:/Dataset 1000'  # Change this to your input directory
output_dir = 'D:/New folder'  # Change this to your output directory
image_size = (224, 224)
batch_size = 16
num_augmented_images = 20000  # Number of augmented images to generate

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize ImageDataGenerator with augmentations
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range= 15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True,
    fill_mode='nearest'
)

# Create a generator for loading images from the input directory
generator = datagen.flow_from_directory(
    input_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,  # We don't need labels for saving augmented images
    save_to_dir=output_dir,
    save_prefix='aug',
    save_format='jpg'
)

# Generate and save the augmented images
total_saved = 0
while total_saved < num_augmented_images:
    batch = next(generator)
    total_saved += len(batch)
    if total_saved >= num_augmented_images:
        break

print(f"Saved {total_saved} augmented images to {output_dir}")
