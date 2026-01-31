"""
FER2013 Data Pipeline
Handles loading, preprocessing, and batching of Facial Expression Recognition (FER2013) dataset.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48
NUM_CLASSES = len(EMOTIONS)
DATASET_PATH = 'fer2013_dataset'


class FERDataPipeline:
    """
    Data pipeline for FER2013 facial expression recognition dataset.
    Handles loading, preprocessing, and batching of images.
    """
    
    def __init__(
        self,
        dataset_path: str = DATASET_PATH,
        img_size: int = IMG_SIZE,
        emotions: list = EMOTIONS,
        val_split: float = 0.2,
        test_split: float = 0.1
    ):
        """
        Initialize the data pipeline.
        
        Args:
            dataset_path: Path to the FER2013 dataset directory
            img_size: Image size (assumed square)
            emotions: List of emotion classes
            val_split: Validation split ratio
            test_split: Test split ratio
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.emotions = emotions
        self.num_classes = len(emotions)
        self.val_split = val_split
        self.test_split = test_split
        self.emotion_to_idx = {e: i for i, e in enumerate(emotions)}
        self.idx_to_emotion = {i: e for i, e in enumerate(emotions)}
        
        # Verify dataset exists
        if not os.path.exists(self.dataset_path):
            logger.warning(f"Dataset not found at {self.dataset_path}")
    
    def _load_images_from_folder(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from a folder.
        
        Args:
            folder_path: Path to the folder containing images
            
        Returns:
            Tuple of (images array, labels array)
        """
        images = []
        labels = []
        
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            return np.array([]), np.array([])
        
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_path = os.path.join(folder_path, emotion)
            if not os.path.exists(emotion_path):
                logger.warning(f"Emotion folder not found: {emotion_path}")
                continue
            
            file_count = 0
            for img_name in os.listdir(emotion_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(emotion_path, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Resize to standard size
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    images.append(img)
                    labels.append(emotion_idx)
                    file_count += 1
                except Exception as e:
                    logger.warning(f"Error loading image {img_path}: {e}")
            
            if file_count > 0:
                logger.info(f"Loaded {file_count} {emotion} images")
        
        return np.array(images), np.array(labels)
    
    def load_data(self, split_data: bool = True) -> Tuple:
        """
        Load and preprocess data from the dataset.
        
        Args:
            split_data: If True, split into train/val/test sets
            
        Returns:
            If split_data=True: (X_train, X_val, X_test, y_train, y_val, y_test)
            If split_data=False: (X, y)
        """
        logger.info("Loading training data...")
        train_path = os.path.join(self.dataset_path, 'train')
        X_train, y_train = self._load_images_from_folder(train_path)
        
        logger.info("Loading test data...")
        test_path = os.path.join(self.dataset_path, 'test')
        X_test, y_test = self._load_images_from_folder(test_path)
        
        if X_train.size == 0 and X_test.size == 0:
            logger.error("No images found in dataset!")
            return None
        
        # Combine train and test for our own split if both exist
        if X_train.size > 0 and X_test.size > 0:
            X_combined = np.concatenate([X_train, X_test])
            y_combined = np.concatenate([y_train, y_test])
        elif X_train.size > 0:
            X_combined = X_train
            y_combined = y_train
        else:
            X_combined = X_test
            y_combined = y_test
        
        # Normalize
        X_combined = X_combined.astype('float32') / 255.0
        X_combined = X_combined.reshape(-1, self.img_size, self.img_size, 1)
        
        # Convert labels to one-hot encoding
        y_combined = to_categorical(y_combined, self.num_classes)
        
        if split_data:
            # Split into train/val/test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_combined, y_combined,
                test_size=self.test_split,
                random_state=42,
                stratify=np.argmax(y_combined, axis=1)
            )
            
            val_split_adjusted = self.val_split / (1 - self.test_split)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_split_adjusted,
                random_state=42,
                stratify=np.argmax(y_temp, axis=1)
            )
            
            logger.info(f"Train set: {X_train.shape[0]} samples")
            logger.info(f"Val set: {X_val.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            logger.info(f"Total samples: {X_combined.shape[0]}")
            return X_combined, y_combined
    
    def get_data_generators(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
        """
        Create data augmentation generators for training.
        
        Args:
            X_train: Training images
            y_train: Training labels
            
        Returns:
            Tuple of (train_generator, augmented_params)
        """
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        return train_datagen, {
            'batch_size': 32,
            'steps_per_epoch': len(X_train) // 32
        }
    
    def create_tf_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset with optional augmentation.
        
        Args:
            X: Images array
            y: Labels array
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply data augmentation
            
        Returns:
            tf.data.Dataset object
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(len(X))
        
        if augment:
            # Simple augmentation using tf.image
            def augment_image(x, y):
                x = tf.image.random_flip_left_right(x)
                x = tf.image.random_rotation(x, 0.1)
                x = tf.image.random_brightness(x, 0.1)
                x = tf.image.random_contrast(x, 0.9, 1.1)
                return x, y
            
            dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_distribution(self, y: np.ndarray) -> dict:
        """
        Get the distribution of classes in the labels.
        
        Args:
            y: One-hot encoded labels
            
        Returns:
            Dictionary with emotion counts
        """
        y_decoded = np.argmax(y, axis=1)
        unique, counts = np.unique(y_decoded, return_counts=True)
        
        distribution = {}
        for idx, count in zip(unique, counts):
            distribution[self.idx_to_emotion[idx]] = int(count)
        
        return distribution


def main():
    """Example usage of the data pipeline."""
    
    # Initialize pipeline
    pipeline = FERDataPipeline()
    
    # Load data with splits
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.load_data(split_data=True)
    
    # Print information
    print(f"\nDataset loaded successfully!")
    print(f"Train set shape: {X_train.shape}")
    print(f"Val set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Class distribution
    print("\nClass distribution in training set:")
    dist = pipeline.get_class_distribution(y_train)
    for emotion, count in dist.items():
        print(f"  {emotion}: {count}")
    
    # Create TensorFlow datasets
    train_dataset = pipeline.create_tf_dataset(
        X_train, y_train,
        batch_size=32,
        shuffle=True,
        augment=True
    )
    
    val_dataset = pipeline.create_tf_dataset(
        X_val, y_val,
        batch_size=32,
        shuffle=False,
        augment=False
    )
    
    test_dataset = pipeline.create_tf_dataset(
        X_test, y_test,
        batch_size=32,
        shuffle=False,
        augment=False
    )
    
    print(f"\nTensorFlow datasets created successfully!")
    print(f"Train dataset: {train_dataset}")
    print(f"Val dataset: {val_dataset}")
    print(f"Test dataset: {test_dataset}")


if __name__ == '__main__':
    main()
