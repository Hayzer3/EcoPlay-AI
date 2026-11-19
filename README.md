# EcoPlay — classificação de resíduos

Serviço de visão computacional para classificar imagens em seis categorias do TrashNet: `cardboard`, `glass`, `metal`, `paper`, `plastic` e `trash`.

## Fluxo implementado

1. `notebooks/01_carregar_e_inspecionar_trashnet.ipynb` carrega o dataset local, confere rótulos e mostra amostras.
2. `scripts/train_classifier.py` separa os dados de forma estratificada (70/15/15), ajusta uma MobileNetV2 pré-treinada, seleciona a melhor época pela perda de validação, mede o resultado no teste e exporta ONNX. O notebook `notebooks/02_treinar_e_avaliar_mobilenetv2.ipynb` conduz a execução no Colab.
3. Opcionalmente, o mesmo script avalia fotos reais rotuladas à parte. As métricas reais não são misturadas às métricas do teste TrashNet.
4. `service/app.py` carrega o ONNX e expõe `POST /classificar` e `GET /health`.

O modelo generativo usado para orientar o descarte pertence à integração Spring AI do projeto Java. Este serviço Python cuida da classificação visual e não substitui essa etapa generativa.

## Dataset

O TrashNet local deve ter uma pasta por classe:

```text
trashnet/dataset-resized/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

A pasta `trashnet/` é ignorada pelo Git. O conjunto atual contém 2.527 imagens. Não envie imagens ou fotos pessoais ao repositório.

Fotos próprias rotuladas podem ser organizadas em `data/raw/real/<classe>/`. Para a avaliação separada, todas as seis subpastas precisam existir e seus nomes devem corresponder às classes do TrashNet. A meta sugerida é reunir 20–30 imagens reais por classe; resultados com amostras pequenas devem ser interpretados com cautela.

## Treinar e avaliar

Use Colab com GPU ou Python 3.11/3.12 local. O ambiente local atual precisa de PyTorch, torchvision e ONNX, instalados a partir de `requirements-train.txt`.

```powershell
python -m pip install -r requirements-train.txt
python scripts/train_classifier.py --data "trashnet/dataset-resized/dataset-resized" --epochs 10
```

Para avaliar também fotos próprias:

```powershell
python scripts/train_classifier.py --data "trashnet/dataset-resized/dataset-resized" --real-data "data/raw/real" --epochs 10
```

Em Colab, coloque o repositório em `MyDrive/EcoPlay/iot-ai` e o ZIP do TrashNet em `MyDrive/EcoPlay/trashnet.zip`, ou ajuste os caminhos na primeira célula de `notebooks/02_treinar_e_avaliar_mobilenetv2.ipynb`. Selecione um runtime com GPU e execute as células em ordem. O notebook copia o ZIP para o armazenamento temporário da sessão, localiza as seis classes e grava os artefatos em `models/` no Drive. Os pesos ImageNet pré-treinados são baixados na primeira execução.

O script grava em `models/`:

- `mobilenetv2_best.pth`: pesos PyTorch da melhor época de validação.
- `mobilenetv2.onnx`: modelo para inferência.
- `metadata.json`: tamanho, normalização e ordem dos rótulos exigidos pela API.
- `evaluation.json`: tamanhos dos conjuntos, acurácia, relatório por classe, matriz de confusão e, quando fornecidas, métricas das fotos reais.

Esses artefatos são locais e ignorados pelo Git. A avaliação não gera números até o treino ser executado. A acurácia do TrashNet não deve ser apresentada como desempenho em fotos reais.

## Executar a API

Instale as dependências do serviço e inicie o servidor a partir da pasta `iot-ai`:

```powershell
python -m pip install -r requirements-service.txt
uvicorn service.app:app --reload
```

Documentação interativa: `http://127.0.0.1:8000/docs`.

O endpoint recebe uma imagem multipart no campo `image`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/classificar" -F "image=@foto.jpg"
```

O JSON de resposta contém a `classe` prevista, `confianca` dessa classe e `probabilidades` para as seis classes. Esses valores são calculados pelo modelo em cada chamada. `confianca` é uma estimativa, não uma garantia de acerto. O modelo só escolhe entre as seis classes conhecidas e não tem classe explícita para objetos fora delas. Use `GET /health` para conferir se os artefatos foram carregados.

## Limitações e uso responsável

- O TrashNet contém imagens em cenários controlados; iluminação, fundo, sujeira, objetos compostos e ângulos diferentes podem reduzir o desempenho no uso real.
- A categoria prevista representa o rótulo visual do dataset, não regras locais de coleta seletiva. A orientação de descarte deve considerar contexto geográfico e regras municipais.
- A avaliação em fotos próprias precisa usar exemplos separados do treino e rótulos revisados por pessoas.
- Não envie fotos com rostos, documentos, placas, endereços ou outros dados pessoais.
