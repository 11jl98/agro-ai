import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


class PlantDiseaseDataLoader:
    
    def __init__(self):
        self.train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=config.VALIDATION_SPLIT,
            fill_mode='nearest'
        )
        
        self.test_datagen = ImageDataGenerator(
            rescale=1.0/255
        )
    
    def load_training_data(self):
        train_generator = self.train_datagen.flow_from_directory(
            str(config.DATASET_DIR),
            target_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=config.RANDOM_SEED
        )
        return train_generator
    
    def load_validation_data(self):
        validation_generator = self.train_datagen.flow_from_directory(
            str(config.DATASET_DIR),
            target_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=config.RANDOM_SEED
        )
        return validation_generator
    
    def load_test_data(self, test_dir):
        test_generator = self.test_datagen.flow_from_directory(
            test_dir,
            target_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        return test_generator
    
    def get_class_names(self, generator):
        class_indices = generator.class_indices
        class_names = {v: k for k, v in class_indices.items()}
        return class_names
