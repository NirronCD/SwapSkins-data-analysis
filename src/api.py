"""
API de preço justo da plataforma SWAP SKINS (CRISP-DM fase 6, Implantação).

Rodar:   uv run uvicorn api:app --app-dir src --reload
Docs:    http://127.0.0.1:8000/docs

Pré-requisito: rodar antes o script 04, que gera models/preco_justo.joblib e
models/snapshot_ultimo_mes.parquet (variáveis mais recentes de cada item).
"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from common import MODELS

artefato = joblib.load(MODELS / "preco_justo.joblib")
modelo, colunas = artefato["modelo"], artefato["colunas"]
FAIXA_BAIXA, FAIXA_ALTA = artefato["faixa_alerta"]
snapshot = pd.read_parquet(MODELS / "snapshot_ultimo_mes.parquet").set_index("item")

app = FastAPI(title="SWAP SKINS - Preço Justo",
              description="Estima o preço justo de uma skin de CS2 e avalia um anúncio.",
              version="1.0.0")


class Consulta(BaseModel):
    item: str = Field(..., examples=["AK-47 | Asiimov (Field-Tested)"],
                      description="market_hash_name exato do item")
    preco_anunciado: float | None = Field(None, gt=0, examples=[10.0],
                                          description="preço do anúncio em US$ (opcional)")


class Resposta(BaseModel):
    item: str
    preco_justo: float
    faixa_aceitavel: tuple[float, float]
    preco_anunciado: float | None
    veredito: str
    modelo: str
    referencia_mes: str


@app.get("/")
def info():
    return {"modelo": artefato["modelo_nome"], "referencia_mes": artefato["referencia_mes"],
            "itens_disponiveis": len(snapshot)}


@app.get("/itens")
def buscar_itens(busca: str = Query(..., min_length=2), limite: int = 20):
    nomes = snapshot.index[snapshot.index.str.contains(busca, case=False, regex=False)]
    return sorted(nomes)[:limite]


@app.post("/preco-justo", response_model=Resposta)
def preco_justo(c: Consulta):
    if c.item not in snapshot.index:
        raise HTTPException(404, "Item não encontrado. Use GET /itens?busca=... para achar o nome.")
    x = snapshot.loc[[c.item], colunas]
    justo = float(10 ** modelo.predict(x)[0])
    faixa = (round(justo * FAIXA_BAIXA, 2), round(justo * FAIXA_ALTA, 2))
    if c.preco_anunciado is None:
        veredito = "sem anúncio para avaliar"
    elif c.preco_anunciado < faixa[0]:
        veredito = "ALERTA: muito abaixo do preço justo (possível golpe ou erro)"
    elif c.preco_anunciado > faixa[1]:
        veredito = "ALERTA: muito acima do preço justo"
    else:
        veredito = "OK: dentro da faixa de mercado"
    return Resposta(item=c.item, preco_justo=round(justo, 2), faixa_aceitavel=faixa,
                    preco_anunciado=c.preco_anunciado, veredito=veredito,
                    modelo=artefato["modelo_nome"], referencia_mes=artefato["referencia_mes"])
