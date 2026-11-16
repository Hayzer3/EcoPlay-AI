# EcoPlay — visão computacional

Esta pasta é o serviço de IA do EcoPlay. Vamos desenvolvê-lo em etapas para que você consiga acompanhar o caminho dos dados até a classificação.

## Plano de trabalho

1. Conferir e organizar o dataset.
2. Preparar treino, validação e teste sem misturar as imagens entre conjuntos.
3. Treinar a MobileNetV2 com Transfer Learning no Colab.
4. Avaliar o modelo no conjunto TrashNet e em fotos reais, separadamente.
5. Exportar o modelo treinado para ONNX.
6. Servir a classificação pelo FastAPI em `POST /classificar`.

Neste primeiro passo, só estamos definindo a estrutura esperada dos dados e um script para contar as imagens. Ainda não há modelo nem métricas.

## Classes e pastas

Coloque as imagens localmente neste formato:

```text
data/raw/
├── trashnet/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── real/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

`trashnet` contém as imagens do TrashNet; `real` contém as fotos próprias da equipe. As pastas usam os nomes originais das classes para evitar ambiguidades no carregamento.

Meta do planejamento: 20–30 fotos reais por classe. O inspetor mostra as quantidades, mas não inventa dados nem considera que a meta foi alcançada.

## Conferir o dataset

Com Python 3.10 ou superior, na pasta `iot-ai`:

```powershell
python scripts/inspect_dataset.py
```

Para conferir outro local:

```powershell
python scripts/inspect_dataset.py --root "D:\dados\ecoplay"
```

O script usa apenas a biblioteca padrão do Python. Ele lista a quantidade de arquivos de imagem por classe e aponta pastas que ainda não existem.

## Cuidados com os dados

- Não envie fotos, datasets ou pesos do modelo para o Git. `.gitignore` já exclui esses artefatos.
- Nas fotos reais, prefira embalagens isoladas em diferentes fundos e condições de luz.
- Evite rostos, documentos, placas, endereços ou qualquer dado pessoal nas fotos.
- Não copie a mesma foto para mais de um conjunto quando fizermos a divisão de treino, validação e teste.
- Métricas só serão registradas depois de treinar e avaliar o modelo com dados identificados.
