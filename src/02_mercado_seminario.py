"""
02 - Análise descritiva do mercado (recria os gráficos do seminário do 1º bimestre).

Responde à pergunta de negócio: o mercado de skins é grande e valorizado o suficiente
para justificar a plataforma SWAP SKINS?

Regras (iguais às do seminário, para manter os resultados reprodutíveis):
- "skin" = canonical_family_key; só versões normais (sem StatTrak);
- exterior representativo por prioridade fixa: FT > MW > FN > WW > BS;
- preço inicial = 1º preço na janela [início, início + 60 dias];
  preço final = último preço na janela [fim - 60 dias, fim];
- preço inicial mínimo de US$ 1 (evita "+900%" de itens de centavos);
- volume mensal normalizado para 30 dias (meses incompletos nas bordas).

Recortes:
- 2013-2026: coorte das skins lançadas em 2013 (as primeiras do jogo), início = estreia do item;
- 2023-2026: itens já existentes em 01/01/2023 (era CS2).
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from common import OUTPUT, connect, save, setup_plots

setup_plots()
con = connect()
FIM = "2026-06-15"
PRIORIDADE_SQL = ("CASE exterior WHEN 'Field-Tested' THEN 1 WHEN 'Minimal Wear' THEN 2 "
                  "WHEN 'Factory New' THEN 3 WHEN 'Well-Worn' THEN 4 ELSE 5 END")

RECORTES = {
    "2013-2026": {"titulo": "Ago 2013 (primeiras skins do jogo) -> Jun 2026",
                  "inicio_item": "f.primeira_data", "filtro": "year(f.estreia_skin) = 2013",
                  "volume_desde": "2013-08-01"},
    "2023-2026": {"titulo": "Jan 2023 (era CS2) -> Jun 2026",
                  "inicio_item": "DATE '2023-01-01'", "filtro": "TRUE",
                  "volume_desde": "2023-01-01"},
}


def valorizacao(cfg: dict) -> pd.DataFrame:
    return con.sql(f"""
        WITH f0 AS (SELECT market_hash_name AS item, canonical_family_key AS skin, exterior,
                           min(date) AS primeira_data
                    FROM prices WHERE NOT stattrak GROUP BY ALL),
        f AS (SELECT *, min(primeira_data) OVER (PARTITION BY skin) AS estreia_skin FROM f0),
        v AS (SELECT f.skin, f.item, f.exterior, f.estreia_skin,
                arg_min(p.price_median, p.date) FILTER (
                    p.date BETWEEN {cfg['inicio_item']} AND {cfg['inicio_item']} + 60) AS preco_inicio,
                arg_max(p.price_median, p.date) FILTER (
                    p.date BETWEEN DATE '{FIM}' - 60 AND DATE '{FIM}') AS preco_fim
              FROM prices p JOIN f ON p.market_hash_name = f.item
              WHERE {cfg['filtro']}
              GROUP BY ALL),
        e AS (SELECT *, (preco_fim / preco_inicio - 1) * 100 AS valorizacao_pct,
                     row_number() OVER (PARTITION BY skin ORDER BY {PRIORIDADE_SQL}) AS rk
              FROM v WHERE preco_inicio >= 1 AND preco_fim IS NOT NULL)
        SELECT skin, item, exterior, preco_inicio, preco_fim, valorizacao_pct
        FROM e WHERE rk = 1 ORDER BY valorizacao_pct DESC""").df()


def serie_indice(itens: list[str], cfg: dict) -> pd.DataFrame:
    lista = ", ".join("'" + i.replace("'", "''") + "'" for i in itens)
    inicio = "DATE '2013-08-14'" if cfg["inicio_item"].startswith("f.") else cfg["inicio_item"]
    df = con.sql(f"""SELECT market_hash_name AS item, date, price_median
                     FROM prices WHERE market_hash_name IN ({lista}) AND date >= {inicio}
                     ORDER BY item, date""").df()
    df["indice"] = df.groupby("item")["price_median"].transform(lambda s: s / s.iloc[0] * 100)
    # mediana móvel de 30 dias: remove picos de um único dia de baixo volume (ruído)
    df["indice"] = df.groupby("item")["indice"].transform(lambda s: s.rolling(30, min_periods=1).median())
    return df


volume_mensal = con.sql("""
    SELECT date_trunc('month', date)::DATE AS mes, sum(volume) AS volume,
           count(DISTINCT date) AS dias, count(DISTINCT market_hash_name) AS itens_rastreados
    FROM prices GROUP BY 1 ORDER BY 1""").df()
volume_mensal["mes"] = pd.to_datetime(volume_mensal["mes"])
volume_mensal["volume_30d"] = volume_mensal["volume"] / volume_mensal["dias"] * 30
volume_mensal["media_movel_3m"] = volume_mensal["volume_30d"].rolling(3, min_periods=1).mean()

for nome, cfg in RECORTES.items():
    print(f"\n=== Recorte {nome} ===")
    out = OUTPUT / "mercado" / nome
    out.mkdir(parents=True, exist_ok=True)

    # ---- valorização top 10 vs mediana ----
    val = valorizacao(cfg)
    mediana = val["valorizacao_pct"].median()
    top10 = val.head(10)
    val.to_csv(out / "valorizacao_skins.csv", index=False)
    print(f"skins elegíveis: {len(val)} | mediana: {mediana:.0f}% | "
          f"maior: {top10.iloc[0]['item']} ({top10.iloc[0]['valorizacao_pct']:.0f}%)")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=top10, y="item", x="valorizacao_pct", color="#4C72B0", ax=ax)
    ax.axvline(mediana, color="#C44E52", ls="--", lw=2, label=f"mediana das {len(val)} skins: {mediana:.0f}%")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f%%", padding=3, fontsize=9)
    ax.set(xlabel="Valorização de preço (%)", ylabel="",
           title=f"Top 10 skins por valorização vs. mediana do mercado\n{cfg['titulo']}")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}%"))
    ax.legend(loc="lower right")
    save(fig, "top10_valorizacao.png", f"mercado/{nome}")

    # ---- índice de preço base 100 (escala log) ----
    idx = serie_indice(top10["item"].tolist(), cfg)
    rotulos = {r.item: f"{r.skin} (+{r.valorizacao_pct:,.0f}%)" for r in top10.itertuples()}
    idx["rotulo"] = idx["item"].map(rotulos)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.lineplot(data=idx, x="date", y="indice", hue="rotulo", hue_order=list(rotulos.values()),
                 lw=1, ax=ax, palette="tab10")
    ax.set_yscale("log")
    ax.axhline(100, color="black", lw=0.8)
    ax.set(xlabel="", ylabel="Índice de preço (início = 100, escala log)\nmediana móvel de 30 dias",
           title=f"Top 10 skins por valorização de preço\n{cfg['titulo']}")
    ax.legend(fontsize=8, loc="upper left", title=None)
    save(fig, "indice_preco_top10.png", f"mercado/{nome}")

    # ---- volume mensal do mercado ----
    vm = volume_mensal[volume_mensal["mes"] >= cfg["volume_desde"]]
    vm.to_csv(out / "volume_mensal.csv", index=False)
    total = vm["volume"].sum()
    print(f"volume total: {total:,.0f} | média mensal (30d): {vm['volume_30d'].mean():,.0f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.bar(vm["mes"], vm["volume_30d"], width=25, color="#4C72B0", alpha=0.85,
            label="Skins vendidas / mês (normalizado para 30 dias)")
    ax1.plot(vm["mes"], vm["media_movel_3m"], color="#C44E52", lw=2, label="Média móvel de 3 meses")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x / 1e6:.0f} mi"))
    ax1.set(ylabel="Unidades vendidas por mês",
            title=f"Volume de vendas do mercado - {nome} (total: {total / 1e9:.2f} bi unidades)")
    ax1.legend(loc="upper left")
    ax2.plot(vm["mes"], vm["itens_rastreados"], color="#55A868")
    ax2.set(ylabel="Itens rastreados")
    save(fig, "volume_mensal.png", f"mercado/{nome}")

    resumo = [f"# Resumo do recorte {nome}", "",
              f"- skins elegíveis para valorização: {len(val)}",
              f"- mediana de valorização: {mediana:.0f}%",
              f"- volume total no período: {total:,.0f} unidades",
              f"- média mensal normalizada (30 dias): {vm['volume_30d'].mean():,.0f}", "",
              "## Top 10", "", top10.to_markdown(index=False, floatfmt=".2f")]
    (out / "resumo.md").write_text("\n".join(resumo), encoding="utf-8")

# ---- skins novas por ano ----
novas = con.sql("""
    SELECT year(estreia) AS ano, count(*) AS skins_novas
    FROM (SELECT canonical_family_key, min(date) AS estreia FROM prices GROUP BY 1)
    GROUP BY 1 ORDER BY 1""").df()
novas.to_csv(OUTPUT / "mercado" / "skins_novas_por_ano.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.barplot(data=novas, x="ano", y="skins_novas", color="#4C72B0", ax=ax)
ax.bar_label(ax.containers[0], padding=2)
ax.set(xlabel="", ylabel="Skins novas", title="Skins de CS:GO/CS2 adicionadas ao jogo, por ano")
save(fig, "skins_novas_por_ano.png", "mercado")

cs2 = volume_mensal[volume_mensal["mes"] >= "2023-01-01"]["volume_30d"].mean()
print(f"\nMédia mensal geral: {volume_mensal['volume_30d'].mean():,.0f} | era CS2: {cs2:,.0f}")
print(f"Itens rastreados: {volume_mensal['itens_rastreados'].iloc[0]} no 1º mês -> "
      f"{volume_mensal['itens_rastreados'].iloc[-1]:,} no último mês")
