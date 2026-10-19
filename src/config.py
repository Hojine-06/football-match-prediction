from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

LEAGUES = {
    "E0": "Premier League", "SP1": "La Liga", "D1": "Bundesliga",
    "I1": "Serie A", "F1": "Ligue 1", "N1": "Eredivisie",
    "P1": "Primeira Liga", "B1": "Pro League",
}
# 2015-2016 -> "1516" ... 2024-2025 -> "2425"
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2015, 2025)]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"