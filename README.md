# Oral Lesion Classification Using Deep Learning with ConvNeXt-Base

Pesos do modelo ConvNeXt-Base treinado para classificação de lesões orais em
quatro categorias clínicas, referente ao artigo *"Oral Lesion Classification
Using Deep Learning with ConvNeXt-Base"* (ENIAC 2026).

## Resultados (conjunto de teste, n=632)

| Métrica | Valor |
|---|---|
| Acurácia | 77,69% |
| MCC | 0,662 |
| AUC-ROC macro | 0,918 |

| Classe | Precisão | Recall | F1 | n |
|---|---|---|---|---|
| Saudável | 87,59% | 95,24% | 91,25% | 126 |
| Benigno | 70,33% | 73,13% | 71,71% | 201 |
| OPMD | 81,44% | 73,38% | 77,20% | 293 |
| OCA | 40,91% | 75,00% | 52,94% | 12 |

## Classes

```
0 = Benign   (Benigno)
1 = Healthy  (Saudável)
2 = OCA      (Câncer bucal)
3 = OPMD     (Potencialmente maligna)
```

## Como carregar o modelo

```python
import keras

model = keras.models.load_model('model_ConvNeXtBase_V2_best.keras')
```

## Como usar para inferência

O modelo espera imagens **224×224×3**, pré-processadas com o `preprocess_input`
específico do ConvNeXt:

```python
import numpy as np
from keras.applications.convnext import preprocess_input
from PIL import Image

img = Image.open('caminho/da/imagem.jpg').convert('RGB').resize((224, 224))
x = np.array(img, dtype=np.float32)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)  # adiciona dimensão de batch

pred = model.predict(x)
classe_predita = np.argmax(pred, axis=1)[0]

classes = ['Benign', 'Healthy', 'OCA', 'OPMD']
print(f"Classe predita: {classes[classe_predita]} ({pred[0][classe_predita]*100:.1f}% de confiança)")
```

## Arquitetura

- **Backbone**: ConvNeXt-Base, pré-treinado no ImageNet, últimas 40 camadas
  ajustadas via *fine-tuning*
- **Classification head**: GlobalAveragePooling2D → BatchNormalization →
  Dropout(0.5) → Dense(512, ReLU) → BatchNormalization → Dropout(0.4) →
  Dense(4, Softmax)

## Dataset de treino

Modelo treinado com o banco de dados de Piyarathne et al. (2024), *"A
Comprehensive Dataset of Annotated Oral Cavity Images for Diagnosis of Oral
Cancer and Oral Potentially Malignant Disorders"*, *Oral Oncology*, 156:106946
— 3.000 imagens de 714 pacientes, com divisão estratificada por paciente
(70% treino / 15% validação / 15% teste).

## ⚠️ Aviso de uso clínico

Este modelo é resultado de pesquisa acadêmica e **não está pronto para
aplicação clínica direta**. A baixa precisão observada na classe OCA (câncer
bucal, n=12 no teste) reflete o tamanho amostral reduzido dessa categoria e
exige cautela na interpretação. Consulte o artigo original para discussão
completa de limitações.

## Citação

Se utilizar este modelo, por favor cite o artigo original (referência
completa disponível após a publicação).
