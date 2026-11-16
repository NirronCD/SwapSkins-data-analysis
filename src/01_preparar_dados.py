"""
01 - Entendimento e preparação dos dados (CRISP-DM fases 2 e 3).

Entrada : data/raw/daily_prices.parquet (10,6 mi linhas diárias, item x dia)
Saídas  : data/processed/painel_mensal.parquet  (item x mês, limpo, base do modelo)
          output/qualidade/relatorio_qualidade.md (números de qualidade citados no artigo)
          output/qualidade/*.csv

O DuckDB é usado só para ler o Parquet e reduzir os dados (dia -> mês) sem estourar a
memória. Toda a limpeza, o tratamento de faltantes e a transformação acontecem em Pandas/NumPy.
"""
import numpy as np
import pandas as pd

from common import OUTPUT, PROCESSED, WEAPON_CLASS, connect

con = connect()
report: list[str] = ["# Relatório de qualidade dos dados\n"]
qdir = OUTPUT / "qualidade"
qdir.mkdir(parents=True, exist_ok=True)


def log(line: str = "") -> None:
    print(line)
    report.append(line)


# ---------------------------------------------------------------------------
# 1. Entendimento dos dados: perfil geral e problemas de qualidade
# ---------------------------------------------------------------------------
perfil = con.sql("""
    SELECT count(*) AS linhas,
           count(DISTINCT market_hash_name)     AS itens,
           count(DISTINCT canonical_family_key) AS skins,
           count(DISTINCT weapon)               AS armas,
           min(date) AS inicio, max(date) AS fim,
           sum(volume) AS volume_total
    FROM prices""").df().iloc[0]
log("## Perfil geral")
for k, v in perfil.items():
    log(f"- {k}: {v:,}" if isinstance(v, (int, np.integer)) else f"- {k}: {v}")

# Colunas constantes não carregam informação nenhuma
cardinalidade = con.sql("""
    SELECT count(DISTINCT source) source, count(DISTINCT currency) currency,
           count(DISTINCT souvenir) souvenir, count(DISTINCT weapon_category) weapon_category,
           count(DISTINCT variant_role) variant_role, count(DISTINCT exterior) exterior
    FROM prices""").df().T.rename(columns={0: "valores_distintos"})
constantes = cardinalidade.index[cardinalidade["valores_distintos"] == 1].tolist()
log(f"\n## Colunas constantes (descartadas): {constantes}")

nulos = con.sql("""
    SELECT count(*) FILTER (price_median IS NULL) AS preco_nulo,
           count(*) FILTER (price_median <= 0)    AS preco_nao_positivo,
           count(*) FILTER (volume IS NULL OR volume = 0) AS volume_nulo_ou_zero,
           count(*) - count(DISTINCT (market_hash_name, date)) AS duplicatas
    FROM prices""").df().iloc[0]
log("\n## Valores nulos / inválidos / duplicados")
for k, v in nulos.items():
    log(f"- {k}: {v:,}")

ruido = con.sql("""
    SELECT avg((source_point_count = 1)::INT) AS pct_spc_1,
           count(*) FILTER (source_point_count = 1) AS linhas_spc_1,
           count(*) FILTER (volume <= 3) AS linhas_volume_ate_3,
           avg((volume <= 3)::INT) AS pct_volume_ate_3
    FROM prices""").df().iloc[0]
log("\n## Ruído (preço diário com pouca informação)")
log(f"- linhas com source_point_count = 1: {ruido.linhas_spc_1:,.0f} ({ruido.pct_spc_1:.1%})")
log(f"- linhas com volume <= 3: {ruido.linhas_volume_ate_3:,.0f} ({ruido.pct_volume_ate_3:.1%})")

# Dados faltantes "invisíveis": dias sem linha dentro da vida de cada item
cobertura = con.sql("""
    SELECT market_hash_name AS item, count(*) AS dias_com_dado,
           date_diff('day', min(date), max(date)) + 1 AS dias_de_vida
    FROM prices GROUP BY 1""").df()
cobertura["cobertura"] = cobertura["dias_com_dado"] / cobertura["dias_de_vida"]
abaixo_90 = (cobertura["cobertura"] < 0.9).sum()
log("\n## Dados faltantes (lacunas nas séries diárias)")
log(f"- itens com cobertura < 90% da própria vida: {abaixo_90} de {len(cobertura)} "
    f"({abaixo_90 / len(cobertura):.1%})")
log(f"- pior cobertura: {cobertura['cobertura'].min():.1%}")
log(f"- cobertura mediana: {cobertura['cobertura'].median():.1%}")
cobertura.sort_values("cobertura").to_csv(qdir / "cobertura_por_item.csv", index=False)

itens_ano = con.sql("""
    SELECT year(date) AS ano, count(DISTINCT market_hash_name) AS itens_rastreados
    FROM prices GROUP BY 1 ORDER BY 1""").df()
itens_ano.to_csv(qdir / "itens_rastreados_por_ano.csv", index=False)
log(f"- itens rastreados: {itens_ano.itens_rastreados.iloc[0]} em {itens_ano.ano.iloc[0]} "
    f"-> {itens_ano.itens_rastreados.iloc[-1]:,} em {itens_ano.ano.iloc[-1]}")

# ---------------------------------------------------------------------------
# 2. Redução dia -> mês (DuckDB) : uma linha por item x mês
#    A mediana mensal dos preços diários já é robusta a picos de um único dia.
# ---------------------------------------------------------------------------
mensal = con.sql("""
    SELECT market_hash_name AS item,
           canonical_family_key AS skin,
           weapon, exterior, stattrak,
           date_trunc('month', date)::DATE AS mes,
           median(price_median) AS preco,
           sum(price_median * volume) / sum(volume) AS preco_medio_ponderado,
           stddev_samp(ln(price_median)) AS volatilidade,
           sum(volume) AS volume,
           count(*) AS dias_com_dado
    FROM prices
    GROUP BY ALL""").df()
log(f"\n## Painel mensal bruto: {len(mensal):,} linhas (item x mês)")

# Dias cobertos pelo dataset em cada mês (ago/2013 tem 18 e jun/2026 tem 15): usado para
# normalizar volume e dias de negociação dos meses parciais das bordas.
dias_mes = con.sql("""SELECT date_trunc('month', date)::DATE AS mes, count(DISTINCT date) AS dias_no_mes
                      FROM prices GROUP BY 1""").df()
dias_mes["mes"] = pd.to_datetime(dias_mes["mes"])

# ---------------------------------------------------------------------------
# 3. Limpeza e transformação (Pandas / NumPy)
# ---------------------------------------------------------------------------
mensal["mes"] = pd.to_datetime(mensal["mes"])
mensal["classe_arma"] = mensal["weapon"].map(WEAPON_CLASS)
assert mensal["classe_arma"].notna().all(), "arma sem classe mapeada"
mensal = mensal.sort_values(["item", "mes"]).reset_index(drop=True)

# 3.1 Completar o calendário: cada item ganha todos os meses entre a 1ª e a última aparição.
#     Meses que não existiam viram NaN explícitos (dado faltante), em vez de sumirem.
atributos = ["skin", "weapon", "exterior", "stattrak", "classe_arma"]
attrs = mensal.groupby("item")[atributos].first()
partes = []
for item, g in mensal.groupby("item", sort=False):
    idx = pd.date_range(g["mes"].min(), g["mes"].max(), freq="MS")
    partes.append(g.set_index("mes").reindex(idx).rename_axis("mes").reset_index().assign(item=item))
painel = pd.concat(partes, ignore_index=True)
painel[atributos] = attrs.loc[painel["item"]].to_numpy()
painel["mes_faltante"] = painel["preco"].isna()
n_falt = int(painel["mes_faltante"].sum())
log(f"- meses faltantes dentro da vida dos itens: {n_falt:,} ({n_falt / len(painel):.2%})")

# 3.2 Outliers mensais: filtro de Hampel no log do preço (janela de 5 meses centrada).
#     Um mês que difere mais que 3x da mediana dos vizinhos é tratado como ruído.
painel["log_preco"] = np.log10(painel["preco"])
mediana_viz = (painel.groupby("item")["log_preco"]
               .transform(lambda s: s.rolling(5, center=True, min_periods=3).median()))
outlier = (painel["log_preco"] - mediana_viz).abs() > np.log10(3)
log(f"- outliers mensais removidos (Hampel, >3x da mediana local): {int(outlier.sum()):,}")
painel.loc[outlier, ["preco", "log_preco"]] = np.nan

# 3.3 Tratamento de faltantes: interpolação linear no log do preço, só para lacunas curtas
#     (até 2 meses seguidos). Lacunas longas ficam sem preço: inventar 6 meses de preço
#     seria criar informação que não existe.
falt_antes = painel["log_preco"].isna()
painel["log_preco"] = (painel.groupby("item")["log_preco"]
                       .transform(lambda s: s.interpolate(limit=2, limit_area="inside")))
painel["preco_imputado"] = falt_antes & painel["log_preco"].notna()
painel["preco"] = 10 ** painel["log_preco"]
painel["volume"] = painel["volume"].fillna(0)            # mês sem linha = nenhuma venda registrada
painel["dias_com_dado"] = painel["dias_com_dado"].fillna(0)
log(f"- preços imputados por interpolação (lacunas <= 2 meses): {int(painel['preco_imputado'].sum()):,}")
log(f"- preços que continuam faltantes (lacunas longas, descartados): {int(painel['log_preco'].isna().sum()):,}")

# 3.4 Normalização temporal: meses parciais (bordas do dataset) viram taxa de 30 dias
painel = painel.merge(dias_mes, on="mes", how="left")
painel["frac_dias_negociados"] = painel["dias_com_dado"] / painel["dias_no_mes"]
painel["volume_30d"] = painel["volume"] / painel["dias_no_mes"] * 30

# 3.5 Confiabilidade: meses com poucos dias de negociação têm preço frágil
#     (< 7 de 30 dias, proporcional nos meses parciais)
painel["baixa_confiabilidade"] = (painel["frac_dias_negociados"] < 7 / 30) & ~painel["preco_imputado"]
log(f"- meses com menos de 7/30 dos dias negociados (baixa confiabilidade): "
    f"{int(painel['baixa_confiabilidade'].sum()):,}")

# 3.6 Variáveis derivadas
primeiro_mes_skin = painel.groupby("skin")["mes"].transform("min")
primeiro_mes_item = painel.groupby("item")["mes"].transform("min")
painel["idade_skin_meses"] = ((painel["mes"].dt.year - primeiro_mes_skin.dt.year) * 12
                              + painel["mes"].dt.month - primeiro_mes_skin.dt.month)
painel["idade_item_meses"] = ((painel["mes"].dt.year - primeiro_mes_item.dt.year) * 12
                              + painel["mes"].dt.month - primeiro_mes_item.dt.month)
painel["ano_lancamento"] = primeiro_mes_skin.dt.year
painel["log_volume"] = np.log10(painel["volume_30d"] + 1)
painel["stattrak"] = painel["stattrak"].astype(bool)

painel = painel.drop(columns=["volatilidade"]).dropna(subset=["log_preco"])
painel.to_parquet(PROCESSED / "painel_mensal.parquet", index=False)
log(f"\n## Painel mensal limpo: {len(painel):,} linhas, {painel['item'].nunique():,} itens, "
    f"{painel['skin'].nunique()} skins")

(qdir / "relatorio_qualidade.md").write_text("\n".join(report), encoding="utf-8")
print(f"\nRelatório salvo em {qdir / 'relatorio_qualidade.md'}")
