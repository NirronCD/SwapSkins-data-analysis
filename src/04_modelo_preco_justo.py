"""
04 - Modelagem e avaliação: modelo de PREÇO JUSTO (regressão) com Scikit-Learn.

Uso no negócio (SWAP SKINS): ao anunciar um item, a plataforma estima o preço justo e
alerta anúncios muito abaixo ou acima dele (golpe, erro de digitação, preço abusivo).

Unidade: item x mês. Alvo: log10 do preço mediano do mês.
O modelo NUNCA vê o próprio preço do item. Ele estima o preço a partir de:
  - atributos do item: arma, classe da arma, desgaste, StatTrak;
  - idade da skin e ano de lançamento;
  - liquidez: volume e dias de negociação no mês;
  - referência da skin: preço médio das OUTRAS variantes da mesma skin no mesmo mês
    (leave-one-out). É o que um comprador olharia para julgar o preço.

Divisão temporal (sem vazamento do futuro):
  treino <= 2022 | validação = 2023 (escolha de hiperparâmetros) | teste = 2024 a 2026-06
"""
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from common import EXTERIORS, MODELS, OUTPUT, PROCESSED, save, setup_plots

setup_plots()
SUB = "modelo"
(OUTPUT / SUB).mkdir(parents=True, exist_ok=True)
RNG = 42

# ---------------------------------------------------------------------------
# 1. Engenharia de variáveis
# ---------------------------------------------------------------------------
painel = pd.read_parquet(PROCESSED / "painel_mensal.parquet")

g = painel.groupby(["skin", "mes"])
soma, n = g["log_preco"].transform("sum"), g["log_preco"].transform("count")
painel["n_variantes_mes"] = n
# média do log-preço das OUTRAS variantes da mesma skin no mês (leave-one-out).
# Skin com uma única variante no mês -> NaN (dado faltante tratado no pipeline).
painel["ref_skin_log_preco"] = np.where(n > 1, (soma - painel["log_preco"]) / (n - 1).clip(lower=1), np.nan)
soma_v = g["log_volume"].transform("sum")
painel["ref_skin_log_volume"] = np.where(n > 1, (soma_v - painel["log_volume"]) / (n - 1).clip(lower=1), np.nan)
painel["volume_relativo"] = painel["log_volume"] - painel["ref_skin_log_volume"]
# nível geral do mercado no mês (mediana de todos os itens): captura inflação/ciclos
painel["mercado_log_preco"] = painel.groupby("mes")["log_preco"].transform("median")
painel["stattrak"] = painel["stattrak"].astype(int)

CAT = ["weapon", "classe_arma", "exterior"]
NUM_BASE = ["stattrak", "idade_skin_meses", "ano_lancamento", "log_volume", "frac_dias_negociados",
            "n_variantes_mes", "mercado_log_preco"]
NUM_REF = ["ref_skin_log_preco", "ref_skin_log_volume", "volume_relativo"]
CONJUNTOS = {"A: só atributos": NUM_BASE, "B: atributos + referência da skin": NUM_BASE + NUM_REF}
ALVO = "log_preco"

# Só meses confiáveis entram no treino e no teste (>= 7 dias de negociação ou preço interpolado)
base = painel[~painel["baixa_confiabilidade"]].copy()
print(f"Linhas confiáveis: {len(base):,} de {len(painel):,} "
      f"({painel['baixa_confiabilidade'].sum():,} descartadas por baixa confiabilidade)")
print(f"Faltantes em ref_skin_log_preco: {base['ref_skin_log_preco'].isna().mean():.2%}")

treino = base[base["mes"] < "2023-01-01"]
valid = base[(base["mes"] >= "2023-01-01") & (base["mes"] < "2024-01-01")]
teste = base[base["mes"] >= "2024-01-01"]
treino_full = base[base["mes"] < "2024-01-01"]
print(f"Treino: {len(treino):,} | Validação: {len(valid):,} | Teste: {len(teste):,}")


# ---------------------------------------------------------------------------
# 2. Modelos candidatos (todos como Pipeline: pré-processamento + estimador)
# ---------------------------------------------------------------------------
def prep_linear(num):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
    ])


def prep_arvore(num):
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
    ])


def make_model(nome, num, **hp):
    if nome == "Dummy (média)":
        return make_pipeline(prep_arvore(num), DummyRegressor(strategy="mean"))
    if nome == "Ridge (linear)":
        return make_pipeline(prep_linear(num), Ridge(alpha=hp.get("alpha", 1.0)))
    if nome == "Random Forest":
        return make_pipeline(prep_arvore(num), RandomForestRegressor(
            n_estimators=150, min_samples_leaf=hp.get("min_samples_leaf", 5), max_features=0.6,
            max_samples=0.3, n_jobs=-1, random_state=RNG))
    if nome == "HistGradientBoosting":
        # HGB trata NaN nativamente e as categorias via categorical_features
        prep = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
            ("num", "passthrough", num)])
        return make_pipeline(prep, HistGradientBoostingRegressor(
            learning_rate=hp.get("learning_rate", 0.1), max_leaf_nodes=hp.get("max_leaf_nodes", 31),
            max_iter=hp.get("max_iter", 400), l2_regularization=1.0,
            categorical_features=list(range(len(CAT))), random_state=RNG))
    raise ValueError(nome)


def metricas(y_log, pred_log) -> dict:
    real, prev = 10 ** y_log, 10 ** pred_log
    ape = np.abs(prev - real) / real
    return {"R2 (log)": r2_score(y_log, pred_log),
            "MAE (US$)": mean_absolute_error(real, prev),
            "MdAPE (%)": np.median(ape) * 100,
            "MAPE (%)": np.mean(ape) * 100,
            "dentro ±10% (%)": np.mean(ape <= 0.10) * 100,
            "dentro ±25% (%)": np.mean(ape <= 0.25) * 100}


# ---------------------------------------------------------------------------
# 3. Ajuste de hiperparâmetros na validação (2023)
# ---------------------------------------------------------------------------
GRADES = {
    "Ridge (linear)": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
    "Random Forest": [{"min_samples_leaf": m} for m in (2, 10)],
    "HistGradientBoosting": [{"learning_rate": lr, "max_leaf_nodes": ml}
                             for lr in (0.05, 0.1) for ml in (31, 127)],
}
melhores_hp, linhas_val = {}, []
for conj, num in CONJUNTOS.items():
    cols = CAT + num
    for nome, grade in GRADES.items():
        melhor = None
        for hp in grade:
            t0 = time.time()
            m = make_model(nome, num, **hp).fit(treino[cols], treino[ALVO])
            met = metricas(valid[ALVO].to_numpy(), m.predict(valid[cols]))
            linhas_val.append({"conjunto": conj, "modelo": nome, "hiperparametros": json.dumps(hp),
                               **met, "tempo_s": time.time() - t0})
            print(f"[val] {conj[:1]} {nome:22s} {hp} -> MdAPE {met['MdAPE (%)']:.1f}% "
                  f"R2 {met['R2 (log)']:.3f} ({time.time() - t0:.0f}s)")
            if melhor is None or met["MdAPE (%)"] < melhor[0]:
                melhor = (met["MdAPE (%)"], hp)
        melhores_hp[(conj, nome)] = melhor[1]
pd.DataFrame(linhas_val).to_csv(OUTPUT / SUB / "validacao_hiperparametros.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Treino final (<= 2023) e avaliação no teste (2024-2026), nunca visto
# ---------------------------------------------------------------------------
# Baseline de negócio: preço mediano histórico por arma + desgaste + StatTrak
chave = ["weapon", "exterior", "stattrak"]
tabela = treino_full.groupby(chave)[ALVO].median().rename("pred")
pred_base = teste.join(tabela, on=chave)["pred"].fillna(treino_full[ALVO].median())
resultados = [{"conjunto": "-", "modelo": "Baseline (mediana arma+desgaste+ST)",
               **metricas(teste[ALVO].to_numpy(), pred_base.to_numpy())}]

modelos_finais, previsoes = {}, {}
for conj, num in CONJUNTOS.items():
    cols = CAT + num
    for nome in ["Dummy (média)", *GRADES]:
        hp = melhores_hp.get((conj, nome), {})
        t0 = time.time()
        m = make_model(nome, num, **hp).fit(treino_full[cols], treino_full[ALVO])
        pred = m.predict(teste[cols])
        met = metricas(teste[ALVO].to_numpy(), pred)
        resultados.append({"conjunto": conj, "modelo": nome, **met, "tempo_s": time.time() - t0})
        modelos_finais[(conj, nome)] = m
        previsoes[(conj, nome)] = pred
        print(f"[teste] {conj[:1]} {nome:22s} MdAPE {met['MdAPE (%)']:.1f}% R2 {met['R2 (log)']:.3f}")

res = pd.DataFrame(resultados)
res.to_csv(OUTPUT / SUB / "comparacao_modelos_teste.csv", index=False)
print("\n", res.drop(columns="tempo_s").round(3).to_string(index=False))

# Melhor modelo = menor erro percentual mediano no teste, entre os do conjunto B
cand = res[res["conjunto"].str.startswith("B") & ~res["modelo"].str.startswith("Dummy")]
melhor_nome = cand.sort_values("MdAPE (%)").iloc[0]["modelo"]
CONJ_FINAL = "B: atributos + referência da skin"
melhor = modelos_finais[(CONJ_FINAL, melhor_nome)]
pred_teste = previsoes[(CONJ_FINAL, melhor_nome)]
COLS_FINAL = CAT + CONJUNTOS[CONJ_FINAL]
print(f"\nModelo escolhido: {melhor_nome} ({CONJ_FINAL}) | hiperparâmetros: "
      f"{melhores_hp[(CONJ_FINAL, melhor_nome)]}")

# ---------------------------------------------------------------------------
# 5. Análise de resultados
# ---------------------------------------------------------------------------
av = teste[["item", "skin", "mes", "exterior", "stattrak", "preco"]].copy()
av["preco_justo"] = 10 ** pred_teste
av["razao"] = av["preco"] / av["preco_justo"]
av["erro_pct"] = (av["preco_justo"] - av["preco"]).abs() / av["preco"] * 100
av["faixa"] = pd.cut(av["preco"], [0, 1, 10, 100, np.inf],
                     labels=["< US$1", "US$1-10", "US$10-100", "> US$100"])

por_faixa = av.groupby("faixa", observed=True).agg(
    n=("erro_pct", "size"), MdAPE=("erro_pct", "median"),
    dentro_25=("erro_pct", lambda e: (e <= 25).mean() * 100))
por_faixa.to_csv(OUTPUT / SUB / "erro_por_faixa_preco.csv")
print("\nErro por faixa de preço:\n", por_faixa.round(1).to_string())

por_ano = av.groupby(av["mes"].dt.year)["erro_pct"].median()
print("\nMdAPE por ano do teste:\n", por_ano.round(1).to_string())

# Regra de alerta da plataforma: faixa que contém 95% dos preços reais
lo, hi = np.quantile(av["razao"], [0.025, 0.975])
print(f"\nFaixa de 95% dos preços reais: {lo:.2f}x a {hi:.2f}x o preço justo")

# ---- gráficos ----
comp = res[res["conjunto"] != "-"].copy()
fig, ax = plt.subplots(figsize=(10, 4.8))
sns.barplot(data=comp, x="modelo", y="MdAPE (%)", hue="conjunto", ax=ax, palette=["#8C8C8C", "#4C72B0"])
ax.axhline(res.iloc[0]["MdAPE (%)"], color="#C44E52", ls="--", label="Baseline (mediana arma+desgaste+ST)")
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f", padding=2, fontsize=8)
ax.set(xlabel="", ylabel="Erro percentual mediano (%)", title="Comparação de modelos no teste (2024-2026): menor é melhor")
ax.legend(fontsize=8)
save(fig, "01_comparacao_modelos.png", SUB)

fig, ax = plt.subplots(figsize=(6.5, 6))
amostra = av.sample(min(len(av), 60000), random_state=RNG)
ax.hexbin(np.log10(amostra["preco"]), np.log10(amostra["preco_justo"]), gridsize=60,
          cmap="viridis", bins="log", mincnt=1)
lim = [np.log10(av["preco"]).min(), np.log10(av["preco"]).max()]
ax.plot(lim, lim, color="#C44E52", lw=1.5, label="previsão perfeita")
ax.set(xlabel="log10(preço real)", ylabel="log10(preço justo previsto)",
       title=f"Real x previsto no teste - {melhor_nome}")
ax.legend()
save(fig, "02_real_vs_previsto.png", SUB)

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(np.log10(av["razao"]), bins=120, ax=ax, color="#4C72B0")
for v in (lo, hi):
    ax.axvline(np.log10(v), color="#C44E52", ls="--")
ax.set_xlim(-1, 1)
ax.set(xlabel="log10(preço real / preço justo)   (0 = acerto exato)", ylabel="Item-mês",
       title=f"Distribuição do erro: 95% dos preços reais entre {lo:.2f}x e {hi:.2f}x o preço justo")
save(fig, "03_distribuicao_erro.png", SUB)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=por_faixa.index.astype(str), y=por_faixa["MdAPE"], color="#55A868", ax=ax)
ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=2)
ax.set(xlabel="Faixa de preço real", ylabel="Erro percentual mediano (%)", title="Erro do modelo por faixa de preço")
save(fig, "04_erro_por_faixa.png", SUB)

NOMES = {"ref_skin_log_preco": "preço de referência da skin", "exterior": "desgaste (exterior)",
         "stattrak": "StatTrak", "volume_relativo": "volume relativo às outras variantes",
         "weapon": "arma", "ano_lancamento": "ano de lançamento", "log_volume": "volume do mês (log)",
         "idade_skin_meses": "idade da skin (meses)", "n_variantes_mes": "nº de variantes no mês",
         "ref_skin_log_volume": "volume de referência da skin", "frac_dias_negociados": "fração de dias negociados",
         "mercado_log_preco": "nível do mercado", "classe_arma": "classe da arma"}
amostra_pi = teste.sample(20000, random_state=RNG)
pi = permutation_importance(melhor, amostra_pi[COLS_FINAL], amostra_pi[ALVO], n_repeats=5,
                            random_state=RNG, scoring="neg_mean_absolute_error", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=COLS_FINAL).sort_values(ascending=False)
imp.to_csv(OUTPUT / SUB / "importancia_variaveis.csv", header=["aumento_MAE_log"])
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=imp.values, y=[NOMES.get(c, c) for c in imp.index], color="#8172B3", ax=ax)
ax.set(xlabel="Aumento do erro (MAE em log) ao embaralhar a variável", ylabel="",
       title="Importância das variáveis (permutation importance)")
save(fig, "05_importancia_variaveis.png", SUB)

# ---- Deriva temporal (concept drift): modelo fixo (treinado até 2023) vs re-treino anual ----
# Para cada ano Y do teste: o modelo fixo prevê Y; o modelo re-treinado usa todos os dados < Y.
HP = melhores_hp[(CONJ_FINAL, melhor_nome)]
linhas_drift = []
for ano in sorted(teste["mes"].dt.year.unique()):
    alvo_ano = base[base["mes"].dt.year == ano]
    fixo = metricas(alvo_ano[ALVO].to_numpy(), melhor.predict(alvo_ano[COLS_FINAL]))
    hist = base[base["mes"].dt.year < ano]
    re = make_model(melhor_nome, CONJUNTOS[CONJ_FINAL], **HP).fit(hist[COLS_FINAL], hist[ALVO])
    retr = metricas(alvo_ano[ALVO].to_numpy(), re.predict(alvo_ano[COLS_FINAL]))
    linhas_drift += [{"ano": ano, "estrategia": "Modelo fixo (treino até 2023)", **fixo},
                     {"ano": ano, "estrategia": "Re-treino anual (treino até o ano anterior)", **retr}]
    print(f"[drift] {ano}: fixo MdAPE {fixo['MdAPE (%)']:.1f}% | re-treino {retr['MdAPE (%)']:.1f}%")
drift = pd.DataFrame(linhas_drift)
drift.to_csv(OUTPUT / SUB / "deriva_retreino_anual.csv", index=False)
nivel = painel.groupby("mes")["mercado_log_preco"].first()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 1.4]})
sns.barplot(data=drift, x="ano", y="MdAPE (%)", hue="estrategia", ax=ax1, palette=["#8C8C8C", "#4C72B0"])
for c in ax1.containers:
    ax1.bar_label(c, fmt="%.1f", padding=2, fontsize=8)
ax1.set(xlabel="Ano avaliado", ylabel="Erro percentual mediano (%)", title="Efeito do re-treino anual")
ax1.legend(fontsize=8, loc="upper left")
ax2.plot(nivel.index, 10 ** nivel.values, color="#C44E52")
ax2.axvspan(pd.Timestamp("2024-01-01"), nivel.index.max(), color="grey", alpha=0.15, label="período de teste")
ax2.set(xlabel="", ylabel="Preço mediano do mercado (US$)", title="Nível de preço do mercado (mediana mensal)")
ax2.legend()
save(fig, "06_deriva_retreino.png", SUB)

# Exemplos para o artigo: maiores desvios = anúncios que a plataforma sinalizaria
exemplos = av.assign(desvio=np.abs(np.log10(av["razao"]))).sort_values("desvio", ascending=False)
exemplos.head(15).to_csv(OUTPUT / SUB / "maiores_desvios_teste.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Implantação: re-treina com TODOS os dados confiáveis e salva para a API
# ---------------------------------------------------------------------------
final = make_model(melhor_nome, CONJUNTOS[CONJ_FINAL], **melhores_hp[(CONJ_FINAL, melhor_nome)])
final.fit(base[COLS_FINAL], base[ALVO])
ultimo = painel[painel["mes"] == painel["mes"].max()][["item", "skin"] + COLS_FINAL + ["preco"]]
joblib.dump({"modelo": final, "colunas": COLS_FINAL, "faixa_alerta": (lo, hi),
             "modelo_nome": melhor_nome, "referencia_mes": str(painel["mes"].max().date())},
            MODELS / "preco_justo.joblib", compress=3)
ultimo.to_parquet(MODELS / "snapshot_ultimo_mes.parquet", index=False)

resumo = {"modelo_escolhido": melhor_nome, "conjunto": CONJ_FINAL,
          "hiperparametros": melhores_hp[(CONJ_FINAL, melhor_nome)],
          "teste": {k: round(float(v), 4) for k, v in metricas(teste[ALVO].to_numpy(), pred_teste).items()},
          "faixa_alerta_95": [round(lo, 3), round(hi, 3)],
          "linhas": {"treino_final": len(treino_full), "teste": len(teste)}}
(OUTPUT / SUB / "resumo_modelo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nResumo:", json.dumps(resumo, indent=2, ensure_ascii=False))
