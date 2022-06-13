import pandas as pd
import numpy as np
from tqdm import tqdm


def add_before_outcomes(df: pd.DataFrame, n_before: int):
    """Adds the n outcomes of the prior matches of the home and away team."""
    before_cols = [f'home_before_{i+1}' for i in range(n_before)] + [f'away_before_{i+1}' for i in range(n_before)]

    # Create columns and set to NaN
    for col in before_cols:
        df[col] = np.NaN

    def fill_before_outcomes(team_id: int):
        """Fills the before outcomes of a given team."""
        team_match_df = df[(df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)]
        if len(team_match_df) > 0:
            team_match_df['team_role'] = team_match_df['home_team_id'].apply(lambda ht: 'H' if ht == team_id else 'A')
            team_match_df['outcome'] = team_match_df.apply(
                lambda r: 1 if r['result'] == r['team_role'] else 0 if r['result'] == 'D' else -1, axis=1)
            for i in range(n_before):
                n = i+1
                team_match_df[f'before_{n}'] = team_match_df['outcome'].shift(n)

            for i, row in team_match_df.iterrows():
                for col in before_cols:
                    row_role = 'home' if row['team_role'] == 'H' else 'away'
                    col_role = col[:4]
                    if row_role == col_role:
                        df.loc[i, col] = row[col[5:]]

    all_teams = set(df['home_team_id'].unique().tolist() + df['away_team_id'].unique().tolist())
    for team in tqdm(all_teams, desc="Adding before outcomes"):  # for each team, fill the outcomes
        fill_before_outcomes(team)
    return df


def add_form(df):
    """Add the form as a weighted average of the before outcomes where recent outcomes have more weight."""
    all_cols = df.columns.tolist()
    before_cols = [col for col in all_cols if col[5:11] == 'before']
    home_before_cols = [col for col in before_cols if col[:4] == 'home']
    away_before_cols = [col for col in before_cols if col[:4] == 'home']
    df['home_form'] = (df[home_before_cols] * [len(home_before_cols)-i for i in range(len(home_before_cols))]).sum(axis=1)
    df['away_form'] = (df[away_before_cols] * [len(away_before_cols) - i for i in range(len(away_before_cols))]).sum(axis=1)
    return df


def get_bookmaker_pred(df: pd.DataFrame) -> pd.Series:
    """Gets the prediction the bookmaker would have made. Configured for Bet365 as bookmaker."""
    bookmaker_pred = df[['b365_H', 'b365_D', 'b365_A']].idxmin(axis=1).str[-1:]
    return bookmaker_pred


def get_outcome_counts(df_in: pd.DataFrame) -> pd.DataFrame:
    """Counts the number of wins, draws, and losses per home team in the prior n matches where n is the number of
    columns that capture prior results (these columns must already exist in DataFrame."""
    df = df_in.copy()
    sites = ['home', 'away']
    outcome_df = pd.DataFrame()
    for site in sites:
        before_cols = [col for col in df.columns if col[:11] == f'{site}_before']
        wins = (df[before_cols] == 1).sum(axis=1)
        draws = (df[before_cols] == 0).sum(axis=1)
        losses = (df[before_cols] == -1).sum(axis=1)
        outcome_df[f"{site}_wins"] = wins
        outcome_df[f"{site}_draws"] = draws
        outcome_df[f"{site}_losses"] = losses
    return outcome_df


