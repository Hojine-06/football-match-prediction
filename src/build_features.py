import json
import pandas as pd
from config import PROCESSED_DIR

WINDOW = 5          # nombre de matchs précédents pris en compte
MIN_PERIODS = 3     # minimum de matchs passés requis

OVERALL = ["gf", "ga", "points", "win", "draw", "gd"]   # forme globale
VENUE = ["gf", "ga", "points"]                          # forme à domicile / extérieur

TEAM_FEATURES = ([f"{c}_avg{WINDOW}" for c in OVERALL]
                 + [f"{c}_venue{WINDOW}" for c in VENUE]
                 + ["rest_days"])

DIFFS = {
    "diff_points": f"points_avg{WINDOW}",
    "diff_gf": f"gf_avg{WINDOW}",
    "diff_ga": f"ga_avg{WINDOW}",
    "diff_gd": f"gd_avg{WINDOW}",
    "diff_venue_points": f"points_venue{WINDOW}",
    "diff_rest": "rest_days",
}

META = ["date", "league", "season", "home_team", "away_team",
        "home_goals", "away_goals", "result"]


def team_rows(df):
    """Une ligne par équipe et par match."""
    base = ["match_id", "date", "league", "season"]
    home = df[base + ["home_team", "home_goals", "away_goals"]].rename(
        columns={"home_team": "team", "home_goals": "gf", "away_goals": "ga"})
    home["is_home"] = 1
    away = df[base + ["away_team", "away_goals", "home_goals"]].rename(
        columns={"away_team": "team", "away_goals": "gf", "home_goals": "ga"})
    away["is_home"] = 0

    rows = pd.concat([home, away], ignore_index=True)
    rows["win"] = (rows["gf"] > rows["ga"]).astype(int)
    rows["draw"] = (rows["gf"] == rows["ga"]).astype(int)
    rows["points"] = rows["win"] * 3 + rows["draw"]
    rows["gd"] = rows["gf"] - rows["ga"]
    return rows.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def add_rolling(rows):
    # shift(1) : on exclut le match courant -> aucune fuite d'information
    past = lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_PERIODS).mean()

    overall = rows.groupby("team")[OVERALL].transform(past)
    overall.columns = [f"{c}_avg{WINDOW}" for c in OVERALL]

    venue = rows.groupby(["team", "is_home"])[VENUE].transform(past)
    venue.columns = [f"{c}_venue{WINDOW}" for c in VENUE]

    rows = pd.concat([rows, overall, venue], axis=1)
    # jours écoulés depuis le dernier match de l'équipe (date connue avant le match)
    rows["rest_days"] = (rows.groupby("team")["date"].diff().dt.days
                         .clip(upper=14).fillna(7))
    return rows


def main():
    df = pd.read_csv(PROCESSED_DIR / "matches_clean.csv",
                     parse_dates=["date"], dtype={"season": str})
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["match_id"] = range(len(df))

    rows = add_rolling(team_rows(df))
    home = rows[rows.is_home == 1].set_index("match_id")[TEAM_FEATURES].add_prefix("home_")
    away = rows[rows.is_home == 0].set_index("match_id")[TEAM_FEATURES].add_prefix("away_")
    out = df.set_index("match_id").join(home).join(away)

    for name, col in DIFFS.items():
        out[name] = out[f"home_{col}"] - out[f"away_{col}"]

    features = ([f"home_{c}" for c in TEAM_FEATURES]
                + [f"away_{c}" for c in TEAM_FEATURES]
                + list(DIFFS))

    n0 = len(out)
    out = out[META + features].dropna(subset=features)
    out.to_csv(PROCESSED_DIR / "features.csv")
    (PROCESSED_DIR / "feature_columns.json").write_text(json.dumps(features, indent=2))

    print(f"Matchs : {n0} -> {len(out)} conservés ({n0 - len(out)} sans historique suffisant)")
    print(f"Variables : {len(features)}")
    print((out["result"].value_counts(normalize=True) * 100).round(1))


if __name__ == "__main__":
    main()