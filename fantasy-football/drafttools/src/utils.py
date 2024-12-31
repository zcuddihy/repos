import json
from difflib import get_close_matches

import numpy as np
import pandas as pd
import requests


# Function to find the best match for a name in another dataset
def find_best_name_match(name, choices, threshold=0.8):
    matches = get_close_matches(name, choices, n=1, cutoff=threshold)
    return matches[0] if matches else None


def fuzzy_merge_dataframes(
    df_left, df_right, fuzzy_match_column: str, merge_on: list, how: str = "outer"
):
    df_right[fuzzy_match_column] = df_right[fuzzy_match_column].apply(
        lambda x: find_best_name_match(x, df_left[fuzzy_match_column])
    )
    merged = df_left.merge(df_right, left_on=merge_on, right_on=merge_on, how=how)
    return merged


def load_adp_data(n_teams: int, scoring: str, year: int):
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{scoring}?teams={n_teams}&year={year}"
    res = requests.get(url)
    response = json.loads(res.text)
    return response


def load_player_data(league_type: str, n_teams: int, year: int, scoring_settings: dict):
    # Load raw ECR data
    ecr_df = pd.read_csv(f"data/{league_type}_ecr.csv")
    ecr_df.rename(
        columns={
            "RK": "overall_rank",
            "TIERS": "tier",
            "PLAYER NAME": "player",
            "TEAM": "team",
            "POS": "position",
            "BEST": "best_rank",
            "WORST": "worst_rank",
            "AVG.": "avg_rank",
            "STD.DEV": "std_dev_rank",
            "ECR VS. ADP": "ecr_vs_adp",
        },
        inplace=True,
    )
    ecr_df["player"] = np.where(
        ecr_df.position.str.contains("DST"),
        ecr_df["player"].str.split(" ").str[-1],
        ecr_df["player"],
    )
    ecr_df["position"] = ecr_df["position"].replace("\d+", "", regex=True)

    # Load ADP
    adp = pd.DataFrame(load_adp_data(n_teams, league_type, year)["players"])
    adp.rename(columns={"name": "player", "stdev": "adp_stddev"}, inplace=True)
    adp = adp[["player", "position", "team", "adp", "adp_stddev"]]

    # Load raw projections data
    proj_df = pd.read_csv("data/projections.csv")
    proj_df["player"] = proj_df["player"].fillna("")
    proj_df["team"] = np.where(proj_df.team == "LVR", "LV", proj_df.team)
    proj_df.fillna(0, inplace=True)
    # proj_df.drop(columns=['Unnamed: 0', 'id','avg_type', 'injury_status', '
    # injury_details', 'season_year', 'week'], inplace=True)
    proj_df.set_index(["player", "team", "position"], inplace=True)

    # Apply scoring settings to get fantasy points data
    fpts_mean = (
        proj_df[[col for col in proj_df.columns if "_sd" not in col]]
        .mul(pd.Series(scoring_settings), axis=1)
        .sum(axis=1)
        .rename("fpts_mean")
        .astype(float)
    )
    fpts_stddev = proj_df[[col for col in proj_df.columns if "_sd" in col]].copy(
        deep=True
    )
    fpts_stddev.rename(columns=lambda x: str(x)[:-3], inplace=True)
    fpts_stddev = np.sqrt(
        (fpts_stddev.mul(pd.Series(scoring_settings), axis=1) ** 2)
        .sum(axis=1)
        .rename("fpts_stddev")
    )
    fpts_proj = pd.concat([fpts_mean, fpts_stddev], axis=1).reset_index()

    # Merge the datasets by using fuzzy matching
    merged = fuzzy_merge_dataframes(
        fpts_proj, ecr_df, "player", ["player", "position", "team"], how="left"
    )
    merged = merged.query("position == @POSITIONS")

    # Clean the datasets to match with Sleeper
    merged["team"] = np.where(merged.team == "JAC", "JAX", merged.team)
    merged["player"] = merged.player.str.replace(" Jr.", "")
    merged = fuzzy_merge_dataframes(
        merged, adp, "player", ["player", "position", "team"], how="inner"
    )

    return merged


def add_drafted_status(player_data: pd.DataFrame, drafted_players: list):
    pd.set_option("future.no_silent_downcasting", True)
    if len(drafted_players) > 0:
        df_drafted_players = pd.DataFrame(drafted_players)
        df_drafted_players["drafted"] = True
        # Merge the datasets by using fuzzy matching
        merged = fuzzy_merge_dataframes(
            player_data, df_drafted_players, "player", ["player", "position", "team"]
        )
        for column in ["drafted", "is_user_draft_pick"]:
            merged[column] = merged[column].fillna(False)
            merged[column] = merged[column].astype(bool)
    else:
        merged = player_data.copy()
        merged["is_user_draft_pick"] = False
        merged["drafted"] = False
        merged["picked_at"] = np.nan
    merged = merged[merged.position.isin(["QB", "RB", "WR", "TE"])].copy(deep=True)
    return merged
