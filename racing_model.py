from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def parse_last_3_positions(value) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return [99, 99, 99]
    parsed = []
    for raw in str(value).replace(" ", "").split(",")[:3]:
        try:
            parsed.append(int(float(raw)))
        except ValueError:
            parsed.append(99)
    return (parsed + [99, 99, 99])[:3]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = [
        "race_number", "distance_m", "draw", "weight_kg", "age",
        "days_since_last_run", "runs_last_90_days", "wins_last_12_months",
        "places_last_12_months", "starts_last_12_months",
        "last_win_days_ago", "official_rating", "market_odds"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pos = df["last_3_positions"].apply(parse_last_3_positions)
    df["last_finish"] = [p[0] for p in pos]
    df["second_last_finish"] = [p[1] for p in pos]
    df["third_last_finish"] = [p[2] for p in pos]
    df["recent_form_score"] = (
        (4 - df["last_finish"].clip(upper=4)) * 3
        + (4 - df["second_last_finish"].clip(upper=4)) * 2
        + (4 - df["third_last_finish"].clip(upper=4))
    )
    df["win_rate_12m"] = df["wins_last_12_months"] / df["starts_last_12_months"].replace(0, np.nan)
    df["market_odds"] = df["market_odds"].clip(lower=1.01)
    df["implied_probability"] = 1.0 / df["market_odds"]
    df["market_rank"] = df.groupby("race_id")["market_odds"].rank(method="min")
    return df

def train_and_predict(*args, **kwargs):
    raise NotImplementedError("Historical CSV disabled.")

@dataclass
class Selection:
    race_id: str
    race_number: int
    course: str
    selection_type: str
    horse_name: str
    model_win_probability: float
    market_odds: float
    model_edge: float
    rank: int
    is_best_selection: bool

def choose_selections(predictions: pd.DataFrame, value_edge_threshold: float = 0.03, min_value_bet_odds: float = 2.0) -> list[Selection]:
    selections: list[Selection] = []
    for race_id, race in predictions.groupby("race_id", sort=False):
        race = race.sort_values(by=["model_win_probability", "market_odds"], ascending=[False, True]).reset_index(drop=True)
        if race.empty:
            continue
        rows = [r for _, r in race.iterrows()]
        pw = rows[0]
        r_num, r_course = int(pw["race_number"]), str(pw["course"])
        selections.append(Selection(race_id, r_num, r_course, "Probable Winner", str(pw["horse_name"]), float(pw["model_win_probability"]), float(pw["market_odds"]), float(pw["model_edge"]), 1, True))

        val_df = race[(race["model_edge"] >= value_edge_threshold) & (race["market_odds"] >= min_value_bet_odds)].sort_values(by="model_edge", ascending=False)
        vb = next(iter(r for _, r in val_df.iterrows()), None)
        if vb is not None:
            selections.append(Selection(race_id, r_num, r_course, "Value Bet", str(vb["horse_name"]), float(vb["model_win_probability"]), float(vb["market_odds"]), float(vb["model_edge"]), 2, False))

        danger_list = [r for r in rows[1:] if vb is None or r["horse_name"] != vb["horse_name"]]
        dh = rows[-1] if not danger_list else danger_list[0]
        selections.append(Selection(race_id, r_num, r_course, "Danger Horse", str(dh["horse_name"]), float(dh["model_win_probability"]), float(dh["market_odds"]), float(dh["model_edge"]), 3, False))
    return selections
