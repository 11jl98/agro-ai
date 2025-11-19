import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


class PlantDiseaseDataLoader:
    
    def __init__(self):
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.5, 1.5],
            channel_shift_range=30.0,
            fill_mode='nearest',
            validation_split=config.VALIDATION_SPLIT
        )
        
        self.test_datagen = ImageDataGenerator(
            rescale=1.0/255
        )
    
    def load_training_data(self):
        return self.train_datagen.flow_from_directory(
            str(config.DATASET_DIR),
            target_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True,
        )
         
    
    def load_validation_data(self):
        return self.train_datagen.flow_from_directory(
            str(config.DATASET_DIR),
            target_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
        )
    
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
        return {v: k for k, v in class_indices.items()}
