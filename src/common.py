"""Constantes e utilitários compartilhados pelos scripts do projeto SWAP SKINS."""
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")  # gera PNG sem precisar de janela gráfica
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW_PARQUET = ROOT / "data" / "raw" / "daily_prices.parquet"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"

for _d in (PROCESSED, OUTPUT, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# Ordem de desgaste, do mais novo para o mais gasto
EXTERIORS = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"]

# Prioridade fixa (determinística) de exterior para representar uma skin.
# Field-Tested primeiro, por ser o desgaste mais negociado (decisão do 1º bimestre).
EXTERIOR_PRIORITY = {"Field-Tested": 1, "Minimal Wear": 2, "Factory New": 3,
                     "Well-Worn": 4, "Battle-Scarred": 5}

# weapon_category no dataset é constante ("Weapon"), então a categoria é derivada aqui.
WEAPON_CLASS = {
    **dict.fromkeys(["AK-47", "AUG", "FAMAS", "Galil AR", "M4A1-S", "M4A4", "SG 553"], "Rifle"),
    **dict.fromkeys(["AWP", "G3SG1", "SCAR-20", "SSG 08"], "Sniper"),
    **dict.fromkeys(["CZ75-Auto", "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18",
                     "P2000", "P250", "R8 Revolver", "Tec-9", "USP-S"], "Pistola"),
    **dict.fromkeys(["MAC-10", "MP5-SD", "MP7", "MP9", "P90", "PP-Bizon", "UMP-45"], "SMG"),
    **dict.fromkeys(["M249", "MAG-7", "Negev", "Nova", "Sawed-Off", "XM1014"], "Pesada"),
}


def connect() -> duckdb.DuckDBPyConnection:
    """Conexão DuckDB com a view `prices` apontando para o Parquet bruto."""
    if not RAW_PARQUET.exists():
        raise SystemExit(f"Dataset não encontrado em {RAW_PARQUET}. Veja o README para baixar.")
    con = duckdb.connect()
    con.execute(f"CREATE VIEW prices AS SELECT * FROM read_parquet('{RAW_PARQUET.as_posix()}')")
    return con


def setup_plots() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name: str, subdir: str = "") -> Path:
    out = OUTPUT / subdir if subdir else OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  salvo: {path.relative_to(ROOT)}")
    return path
