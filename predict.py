import sys
import numpy as np
import json
from pathlib import Path
from PIL import Image
import config
from model import PlantDiseaseClassifier
from image_preprocessor import ImagePreprocessor


class DiseasePredictor:
    
    def __init__(self, model_path=None, class_mapping_path=None):
        self.classifier = PlantDiseaseClassifier()
        
        model_file = model_path or config.CHECKPOINT_PATH
        self.classifier.load_model(model_file)
        
        mapping_file = class_mapping_path or (config.MODEL_DIR / "class_mapping.json")
        with open(mapping_file, 'r') as f:
            self.class_names = json.load(f)
        
        self.class_names = {int(k): v for k, v in self.class_names.items()}
    
    def preprocess_image(self, image_path, use_advanced_preprocessing=True):
        if use_advanced_preprocessing:
            image_array = ImagePreprocessor.preprocess_real_world_image(
                image_path, 
                apply_background_removal=True
            )
        else:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(config.IMAGE_SIZE)
            image_array = np.array(image) / 255.0
        
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    def predict(self, image_path, top_k=3, use_advanced_preprocessing=True):
        preprocessed_image = self.preprocess_image(image_path, use_advanced_preprocessing)
        predictions = self.classifier.model.predict(preprocessed_image, verbose=0)
        
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_probabilities = predictions[0][top_indices]
        
        results = []
        for idx, prob in zip(top_indices, top_probabilities):
            disease_name = self.class_names[idx]
            confidence = float(prob * 100)
            results.append({
                'disease': disease_name,
                'confidence': confidence
            })
        
        return results
    
    def predict_batch(self, image_paths, top_k=3, use_advanced_preprocessing=True):
        batch_results = []
        for image_path in image_paths:
            results = self.predict(image_path, top_k, use_advanced_preprocessing)
            batch_results.append({
                'image': str(image_path),
                'predictions': results
            })
        return batch_results
    
    def print_prediction(self, results):
        print("\n" + "="*60)
        print("DIAGNÓSTICO DE DOENÇA DA PLANTA")
        print("="*60 + "\n")
        
        for i, result in enumerate(results, 1):
            disease = result['disease'].replace('___', ' - ')
            confidence = result['confidence']
            
            print(f"{i}. {disease}")
            print(f"   Confiança: {confidence:.2f}%")
            print()
        
        print("="*60 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Uso: python predict.py <caminho_da_imagem> [--simple]")
        print("\nOpções:")
        print("  --simple: Desabilita pré-processamento avançado")
        return
    
    image_path = sys.argv[1]
    use_advanced = '--simple' not in sys.argv
    
    if not Path(image_path).exists():
        print(f"Erro: Imagem não encontrada em {image_path}")
        return
    
    print("Carregando modelo...")
    predictor = DiseasePredictor()
    
    preprocessing_mode = "avançado" if use_advanced else "simples"
    print(f"Analisando imagem com pré-processamento {preprocessing_mode}: {image_path}")
    
    results = predictor.predict(image_path, top_k=3, use_advanced_preprocessing=use_advanced)
    predictor.print_prediction(results)


if __name__ == "__main__":
    main()