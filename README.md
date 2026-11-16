# SWAP SKINS: mineração de dados do mercado de skins de CS2

Projeto de Mineração de Dados (CRISP-DM) para avaliar a viabilidade da **SWAP SKINS**, uma
plataforma web de troca e venda P2P de skins de Counter-Strike 2, e construir o seu
primeiro componente inteligente: um **modelo de preço justo** que alerta anúncios fora do
preço de mercado (golpes, erros, preços abusivos).

## Dados

[CS2 Skin Price History 2013-2026](https://www.kaggle.com/datasets/devynpruden/cs2-skin-price-history-2013-2026)
(Kaggle, Apache 2.0): 10,6 milhões de registros diários (item x dia) do Steam Community Market,
de 14/08/2013 a 15/06/2026, com 4.342 itens e 521 skins.

O dataset **não está no repositório** (284 MB). Para baixar:

```bash
mkdir -p data/raw && cd data/raw
curl -L -o dataset.zip "https://www.kaggle.com/api/v1/datasets/download/devynpruden/cs2-skin-price-history-2013-2026"
unzip dataset.zip && rm dataset.zip
```

## Como rodar

Requer [uv](https://docs.astral.sh/uv/). Ele instala o Python e todas as dependências nas
versões travadas em `uv.lock`.

```bash
uv sync
uv run python src/01_preparar_dados.py        # qualidade + limpeza -> data/processed/painel_mensal.parquet
uv run python src/02_mercado_seminario.py     # gráficos do mercado (seminário do 1º bimestre)
uv run python src/03_analise_exploratoria.py  # EDA com Seaborn
uv run python src/04_modelo_preco_justo.py    # treino, comparação e avaliação dos modelos
uv run uvicorn api:app --app-dir src          # API em http://127.0.0.1:8000/docs
```

## Estrutura (fases do CRISP-DM)

| Script | Fase CRISP-DM | O que faz |
|---|---|---|
| `src/01_preparar_dados.py` | Entendimento e Preparação dos dados | Perfil e relatório de qualidade; reduz dia -> mês (DuckDB); em Pandas: completa o calendário, remove outliers (filtro de Hampel), interpola lacunas curtas, marca meses de baixa confiabilidade, cria variáveis derivadas |
| `src/02_mercado_seminario.py` | Entendimento do negócio / descritivo | Volume mensal normalizado, top 10 valorização vs. mediana, índice base 100, skins novas por ano |
| `src/03_analise_exploratoria.py` | Entendimento dos dados | Distribuições, preço por desgaste/StatTrak/classe, correlação, prêmio StatTrak, cobertura |
| `src/04_modelo_preco_justo.py` | Modelagem e Avaliação | Ridge, Random Forest e HistGradientBoosting vs. baselines; divisão temporal; métricas; importância das variáveis |
| `src/api.py` | Implantação | FastAPI: `POST /preco-justo` devolve preço justo, faixa aceitável e veredito do anúncio |

Saídas em `output/` (`qualidade/`, `mercado/`, `eda/`, `modelo/`): gráficos PNG, CSVs e resumos.

## Exemplo de uso da API

```bash
curl -X POST http://127.0.0.1:8000/preco-justo -H "Content-Type: application/json" \
     -d '{"item": "AK-47 | Asiimov (Field-Tested)", "preco_anunciado": 10.0}'
```
