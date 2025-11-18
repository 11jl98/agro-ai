# AgroAI - Diagnóstico de Doenças em Plantas

Sistema de Deep Learning para detecção e classificação de doenças e pragas em plantas usando Redes Neurais Convolucionais (CNN).

## Dataset

Este projeto utiliza o **PlantVillage Dataset**, que contém 38 classes diferentes de doenças em plantas, incluindo:
- Tomate (10 classes)
- Batata (3 classes)
- Pimentão (2 classes)
- Milho (4 classes)
- Uva (4 classes)
- Entre outras culturas

Para baixar o dataset:
1. Acesse: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Baixe e extraia para `data/plantvillage/`
3. A estrutura deve ser: `data/plantvillage/<nome_da_classe>/<imagens>`

## Estrutura do Projeto

```
agroAi/
├── config.py              # Configurações e hiperparâmetros
├── data_loader.py         # Carregamento e preprocessamento de dados
├── model.py               # Arquitetura da rede neural
├── train.py               # Script de treinamento
├── predict.py             # Script de predição
├── requirements.txt       # Dependências do projeto
├── data/                  # Diretório de dados
│   └── plantvillage/      # Dataset (baixar separadamente)
├── models/                # Modelos treinados
└── results/               # Históricos e gráficos
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

### Treinamento

```bash
python train.py
```

O treinamento irá:
- Carregar e preprocessar as imagens
- Treinar o modelo por até 50 épocas (com early stopping)
- Salvar o melhor modelo em `models/plant_disease_classifier.keras`
- Gerar gráficos de acurácia e loss em `results/`

### Predição

```bash
python predict.py caminho/para/imagem.jpg
```

Exemplo:
```bash
python predict.py data/test_images/tomato_leaf.jpg
```

O sistema retornará as 3 previsões mais prováveis com suas respectivas confianças.

## Arquitetura do Modelo

A CNN possui:
- 4 blocos convolucionais (32, 64, 128, 256 filtros)
- Batch Normalization após cada camada
- MaxPooling e Dropout para regularização
- 2 camadas densas (512, 256 neurônios)
- Camada de saída com 38 neurônios (softmax)

## Hiperparâmetros

- Tamanho de imagem: 224x224
- Batch size: 32
- Learning rate: 0.001
- Épocas máximas: 50
- Validação: 20% dos dados
- Otimizador: Adam

## Métricas

- Acurácia (accuracy)
- Top-3 Acurácia (top_3_accuracy)
- Loss (categorical crossentropy)

## Resultados Esperados

Com o dataset PlantVillage completo, espera-se:
- Acurácia de validação: ~95-98%
- Top-3 acurácia: ~99%

## Uso em Produção

Para usar o modelo treinado:

```python
from predict import DiseasePredictor

predictor = DiseasePredictor()
results = predictor.predict("imagem.jpg", top_k=3)

for result in results:
    print(f"{result['disease']}: {result['confidence']:.2f}%")
```

## Licença

MIT
