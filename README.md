# EcoPlay — classificação de resíduos

Serviço de visão computacional para classificar imagens em seis categorias do TrashNet: `cardboard`, `glass`, `metal`, `paper`, `plastic` e `trash`.

## Fluxo implementado

1. `notebooks/01_carregar_e_inspecionar_trashnet.ipynb` carrega o dataset local, confere rótulos e mostra amostras.
2. `scripts/train_classifier.py` separa os dados de forma estratificada (70/15/15), ajusta uma MobileNetV2 pré-treinada, seleciona a melhor época pela perda de validação, mede o resultado no teste e exporta ONNX. O notebook `notebooks/02_treinar_e_avaliar_mobilenetv2.ipynb` conduz a execução no Jupyter local.
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

Jupyter local é suficiente; GPU é opcional. Python 3.11 ou 3.12 é recomendado. Crie um ambiente virtual para o treino, instale as dependências e registre um kernel Jupyter:

```powershell
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
python -m ipykernel install --user --name ecoplay-ai --display-name "Python (EcoPlay AI)"
```

No VS Code/Jupyter, selecione o kernel **Python (EcoPlay AI)**, abra `notebooks/02_treinar_e_avaliar_mobilenetv2.ipynb` e execute as células em ordem. O notebook localiza a pasta do projeto e o dataset já extraído. Se a máquina tiver GPU compatível com CUDA, o script a utiliza; caso contrário, treina em CPU, o que pode demorar mais.

O treino também pode ser iniciado sem notebook:

```powershell
python scripts/train_classifier.py --data "trashnet/dataset-resized/dataset-resized" --epochs 10
```

Para avaliar também fotos próprias:

```powershell
python scripts/train_classifier.py --data "trashnet/dataset-resized/dataset-resized" --real-data "data/raw/real" --epochs 10
```

Na primeira execução, os pesos ImageNet pré-treinados são baixados. A pasta local `trashnet/` deve estar disponível para o treino.

O script grava em `models/`:

- `mobilenetv2_best.pth`: pesos PyTorch da melhor época de validação.
- `mobilenetv2.onnx`: modelo para inferência.
- `metadata.json`: tamanho, normalização e ordem dos rótulos exigidos pela API.
- `evaluation.json`: tamanhos dos conjuntos, acurácia, relatório por classe, matriz de confusão e, quando fornecidas, métricas das fotos reais.

`mobilenetv2.onnx` e `metadata.json` estão versionados no repositório para que a API consiga iniciar no deploy. O checkpoint PyTorch `mobilenetv2_best.pth` e o relatório completo `evaluation.json` são artefatos locais ignorados pelo Git.

### Resultado registrado no conjunto de teste

Na execução registrada, a divisão estratificada com semente 42 usou 1.769 imagens para treino, 379 para validação e 379 para teste. No teste reservado do TrashNet, o modelo obteve **90,24% de acurácia** e **90,01% de F1 macro**. Esses valores medem apenas imagens do TrashNet; não houve avaliação com fotos reais (`real_photos` não foi avaliado). O arquivo local `models/evaluation.json` contém o relatório por classe e a matriz de confusão. A acurácia do TrashNet não deve ser apresentada como desempenho em fotos reais.

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
