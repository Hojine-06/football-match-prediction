import numpy as np
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR, LEAGUES

KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF"]
RENAME = {
    "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_on_target", "AST": "away_shots_on_target",
    "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls",
    "League": "league", "Season": "season",
}

def load_file(path):
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(how="all").dropna(subset=["HomeTeam", "AwayTeam"])
    code, season = path.stem.split("_")
    df = df[[c for c in KEEP if c in df.columns]].copy()
    df["League"] = LEAGUES[code]
    df["Season"] = season
    return df

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = [load_file(p) for p in sorted(RAW_DIR.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
    n0 = len(df)

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce", format="mixed")
    for c in df.columns:
        if c not in ["date", "home_team", "away_team", "result", "league", "season"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "home_goals", "away_goals", "result"])
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])

    expected = np.where(df.home_goals > df.away_goals, "H",
                np.where(df.home_goals < df.away_goals, "A", "D"))
    incoherent = (df.result != expected).sum()
    df = df[df.result == expected]

    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(PROCESSED_DIR / "matches_clean.csv", index=False)

    print(f"Lignes brutes : {n0} | conservées : {len(df)} | incohérentes : {incoherent}")
    print("Valeurs manquantes (%) :")
    print((df.isna().mean() * 100).round(1))

if __name__ == "__main__":
    main()