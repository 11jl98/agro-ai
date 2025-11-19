import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
import json
from datetime import datetime
import config
from data_loader import PlantDiseaseDataLoader
from model import PlantDiseaseClassifier


class ResumeCheckpoint(Callback):
    
    def __init__(self, checkpoint_path, resume_val_accuracy=None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.best_val_accuracy = resume_val_accuracy or 0.0
        self.verbose = 1
    
    def on_epoch_end(self, logs=None):
        current_val_accuracy = logs.get('val_accuracy', 0)
        
        if current_val_accuracy > self.best_val_accuracy:
            if self.verbose > 0:
                print(f"\nval_accuracy improved from {self.best_val_accuracy:.5f} to {current_val_accuracy:.5f}, saving model")
            
            self.model.save(self.checkpoint_path)
            self.best_val_accuracy = current_val_accuracy
        else:
            if self.verbose > 0:
                print(f"\nval_accuracy did not improve from {self.best_val_accuracy:.5f}")


class ModelTrainer:
    
    def __init__(self, resume=False, resume_path=None, initial_epoch=0, resume_val_accuracy=None):
        self.data_loader = PlantDiseaseDataLoader()
        self.classifier = PlantDiseaseClassifier()
        self.history = None
        self.class_names = None
        self.resume = resume
        self.resume_path = resume_path
        self.initial_epoch = initial_epoch
        self.resume_val_accuracy = resume_val_accuracy
    
    def prepare_data(self):
        train_generator = self.data_loader.load_training_data()
        validation_generator = self.data_loader.load_validation_data()
        self.class_names = self.data_loader.get_class_names(train_generator)
        
        return train_generator, validation_generator
    
    def setup_callbacks(self):
        if self.resume and self.resume_val_accuracy is not None:
            checkpoint = ResumeCheckpoint(
                checkpoint_path=str(config.CHECKPOINT_PATH),
                resume_val_accuracy=self.resume_val_accuracy
            )
        else:
            checkpoint = ModelCheckpoint(
                str(config.CHECKPOINT_PATH),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        return [checkpoint, early_stopping, reduce_lr]
    
    def train(self):
        print("Carregando dados...")
        train_generator, validation_generator = self.prepare_data()
        
        print(f"\nTotal de classes: {len(self.class_names)}")
        print(f"Imagens de treinamento: {train_generator.samples}")
        print(f"Imagens de validação: {validation_generator.samples}")
        
        if self.resume and self.resume_path:
            print(f"\nRetomando treinamento do modelo: {self.resume_path}")
            self.classifier.load_for_training(self.resume_path)
            print(f"Continuando da época: {self.initial_epoch + 1}")
            if self.resume_val_accuracy:
                print(f"Melhor val_accuracy anterior: {self.resume_val_accuracy:.5f}")
        else:
            print("\nConstruindo modelo...")
            self.classifier.build_model(num_classes=len(self.class_names))
            self.classifier.compile_model()

        self.classifier.get_summary()
        
        print("\nIniciando treinamento...")
        callbacks = self.setup_callbacks()
        
        self.history = self.classifier.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=config.EPOCHS,
            initial_epoch=self.initial_epoch,
            callbacks=callbacks,
            verbose=1
        )
         
        print("\nTreinamento concluído!")
        return self.history
    
    def save_training_history(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = config.RESULTS_DIR / f"training_history_{timestamp}.json"
        
        history_dict = {
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
            'loss': [float(x) for x in self.history.history['loss']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'top_3_accuracy': [float(x) for x in self.history.history['top_3_accuracy']],
            'val_top_3_accuracy': [float(x) for x in self.history.history['val_top_3_accuracy']]
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"Histórico salvo em: {history_path}")
    
    def save_class_mapping(self):
        class_mapping_path = config.MODEL_DIR / "class_mapping.json"
        with open(class_mapping_path, 'w') as f:
            json.dump(self.class_names, f, indent=4)
        print(f"Mapeamento de classes salvo em: {class_mapping_path}")
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self.history.history['accuracy'], label='Treino')
        axes[0].plot(self.history.history['val_accuracy'], label='Validação')
        axes[0].set_title('Acurácia do Modelo')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Acurácia')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['loss'], label='Treino')
        axes[1].plot(self.history.history['val_loss'], label='Validação')
        axes[1].set_title('Loss do Modelo')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = config.RESULTS_DIR / f"training_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {plot_path}")
        plt.close()


def main():
    tf.random.set_seed(config.RANDOM_SEED)
    
    trainer = ModelTrainer(
        resume=config.RESUME_TRAINING,
        resume_path=config.RESUME_MODEL_PATH,
        initial_epoch=config.INITIAL_EPOCH,
        resume_val_accuracy=config.RESUME_VAL_ACCURACY
    )
    trainer.train()
    trainer.save_training_history()
    trainer.save_class_mapping()
    trainer.plot_training_history()
    
    print(f"\nModelo final salvo em: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
