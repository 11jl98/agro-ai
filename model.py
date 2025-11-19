import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import config


class PlantDiseaseClassifier:
    
    def __init__(self):
        self.model = None
    
    def build_model(self, num_classes=config.NUM_CLASSES):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(*config.IMAGE_SIZE, 3),
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
    
    def get_summary(self):
        return self.model.summary()
    
    def save_model(self, filepath=None):
        save_path = filepath or config.CHECKPOINT_PATH
        self.model.save(save_path)
        return save_path
    
    def load_model(self, filepath=None):
        load_path = filepath or config.CHECKPOINT_PATH
        self.model = tf.keras.models.load_model(load_path)
        return self.model
    
    def load_for_training(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Modelo carregado de: {model_path}")
        return self.model
