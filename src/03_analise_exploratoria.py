"""
03 - Análise exploratória (EDA) do painel mensal limpo, com Seaborn.

Objetivo: entender o comportamento do preço antes de modelar. Quais atributos explicam
o preço? A distribuição é assimétrica? Onde estão os dados faltantes?
Entrada: data/processed/painel_mensal.parquet (gerado pelo script 01).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import EXTERIORS, OUTPUT, PROCESSED, save, setup_plots

setup_plots()
SUB = "eda"
painel = pd.read_parquet(PROCESSED / "painel_mensal.parquet")
# Recorte transversal: últimos 12 meses completos, para comparar itens sem o efeito do tempo
recente = painel[(painel["mes"] >= "2025-06-01") & (painel["mes"] < "2026-06-01")].copy()
recente["stattrak_txt"] = np.where(recente["stattrak"], "StatTrak", "Normal")
print(f"Painel: {len(painel):,} linhas | recorte recente: {len(recente):,} linhas")

# 1. Distribuição do preço: muito assimétrica -> justifica a transformação log
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
sns.histplot(recente["preco"], bins=80, ax=axes[0], color="#4C72B0")
axes[0].set(title="Preço em US$ (escala original)", xlabel="Preço (US$)", ylabel="Item-mês")
sns.histplot(recente["log_preco"], bins=80, ax=axes[1], color="#55A868", kde=True)
axes[1].set(title="log10(preço): distribuição quase simétrica", xlabel="log10(preço)", ylabel="")
skew_orig, skew_log = recente["preco"].skew(), recente["log_preco"].skew()
fig.suptitle(f"Distribuição de preços (assimetria: {skew_orig:.1f} -> {skew_log:.2f} após log)")
save(fig, "01_distribuicao_preco.png", SUB)

faixas = pd.cut(recente["preco"], [0, 1, 10, 100, np.inf],
                labels=["< US$1", "US$1-10", "US$10-100", "> US$100"])
dist_faixas = faixas.value_counts(normalize=True).sort_index()
print("Distribuição por faixa de preço:\n", dist_faixas.round(3).to_string())
dist_faixas.to_csv(OUTPUT / SUB / "distribuicao_faixas_preco.csv")

# 2. Preço por exterior e StatTrak
fig, ax = plt.subplots(figsize=(10, 4.8))
sns.boxplot(data=recente, x="exterior", y="log_preco", hue="stattrak_txt", order=EXTERIORS,
            showfliers=False, ax=ax, palette=["#4C72B0", "#DD8452"])
ax.set(xlabel="", ylabel="log10(preço)", title="Preço por desgaste (exterior) e StatTrak")
ax.legend(title=None)
save(fig, "02_preco_por_exterior_stattrak.png", SUB)

# 3. Preço por classe de arma
ordem = recente.groupby("classe_arma")["log_preco"].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.boxplot(data=recente, x="classe_arma", y="log_preco", order=ordem, showfliers=False,
            ax=ax, color="#8172B3")
ax.set(xlabel="", ylabel="log10(preço)", title="Preço por classe de arma")
save(fig, "03_preco_por_classe_arma.png", SUB)

# 4. Liquidez x preço: itens caros vendem menos
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.hexbin(recente["log_volume"], recente["log_preco"], gridsize=50, cmap="viridis", mincnt=1,
          bins="log")
ax.set(xlabel="log10(volume mensal + 1)", ylabel="log10(preço)",
       title=f"Volume x preço (correlação: {recente[['log_volume', 'log_preco']].corr().iloc[0, 1]:.2f})")
save(fig, "04_volume_vs_preco.png", SUB)

# 5. Matriz de correlação das variáveis numéricas
num = recente[["log_preco", "log_volume", "frac_dias_negociados", "idade_skin_meses",
               "idade_item_meses", "ano_lancamento"]].assign(stattrak=recente["stattrak"].astype(int))
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, square=True)
ax.set_title("Correlação entre variáveis numéricas (Pearson)")
save(fig, "05_correlacao.png", SUB)

# 6. Prêmio StatTrak por exterior (mesma skin, mesmo mês)
pivo = recente.pivot_table(index=["skin", "exterior", "mes"], columns="stattrak",
                           values="preco", aggfunc="first").dropna()
pivo["premio"] = pivo[True] / pivo[False]
premio = pivo.reset_index().groupby("exterior")["premio"].median().reindex(EXTERIORS)
print("Prêmio StatTrak (mediana, x vezes):\n", premio.round(2).to_string())
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=premio.index, y=premio.values, color="#DD8452", ax=ax)
ax.bar_label(ax.containers[0], fmt="%.2fx", padding=2)
ax.set(xlabel="", ylabel="Preço StatTrak / preço normal", title="Prêmio mediano da versão StatTrak")
save(fig, "06_premio_stattrak.png", SUB)

# 7. Dados faltantes: cobertura diária por item
cob = pd.read_csv(OUTPUT / "qualidade" / "cobertura_por_item.csv")
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(cob["cobertura"] * 100, bins=50, ax=ax, color="#C44E52")
ax.axvline(90, color="black", ls="--", label="limite de 90%")
ax.set_yscale("log")
ax.set(xlabel="Cobertura diária dentro da vida do item (%)", ylabel="Nº de itens (escala log)",
       title=f"Lacunas nas séries: {(cob['cobertura'] < .9).sum()} itens abaixo de 90% de cobertura")
ax.legend()
save(fig, "07_cobertura_itens.png", SUB)

# 8. Dias de negociação por mês (confiabilidade do preço mensal)
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(painel["dias_com_dado"], bins=32, discrete=True, ax=ax, color="#937860")
ax.axvline(6.5, color="black", ls="--", label="< 7 dias: baixa confiabilidade")
ax.set_yscale("log")
ax.set(xlabel="Dias com negociação no mês", ylabel="Item-mês (escala log)",
       title="Quantos dias do mês cada item teve negociação")
ax.legend()
save(fig, "08_dias_negociacao.png", SUB)
