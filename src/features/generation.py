import pandas as pd
import numpy as np
from tqdm import tqdm


def add_before_outcomes(df: pd.DataFrame, n_before: int):
    before_cols = [f'HomeBefore{i+1}' for i in range(n_before)] + [f'AwayBefore{i+1}' for i in range(n_before)]
    for col in before_cols:
        df[col] = np.NaN

    def fill_before_outcomes(team_id: int):
        team_match_df = df[(df['HomeTeamID'] == team_id) | (df['AwayTeamID'] == team_id)]
        if len(team_match_df) > 0:
            team_match_df['TeamRole'] = team_match_df['HomeTeamID'].apply(lambda ht: 'H' if ht == team_id else 'A')
            team_match_df['Outcome'] = team_match_df.apply(
                lambda r: 1 if r['FTR'] == r['TeamRole'] else 0 if r['FTR'] == 'D' else -1, axis=1)
            for i in range(n_before):
                n = i+1
                team_match_df[f'Before{n}'] = team_match_df['Outcome'].shift(n)

            for i, row in team_match_df.iterrows():
                for col in before_cols:
                    row_role = 'Home' if row['TeamRole'] == 'H' else 'Away'
                    col_role = col[:4]
                    if row_role == col_role:
                        df.loc[i, col] = row[col[4:]]

    all_teams = set(df['HomeTeamID'].unique().tolist() + df['AwayTeamID'].unique().tolist())
    for team in tqdm(all_teams, desc="Adding before outcomes"):
        fill_before_outcomes(team)
    return df


def add_form(df):
    all_cols = df.columns.tolist()
    before_cols = [col for col in all_cols if col[4:10] == 'Before']
    home_before_cols = [col for col in before_cols if col[:4] == 'Home']
    away_before_cols = [col for col in before_cols if col[:4] == 'Away']
    df['HomeForm'] = (df[home_before_cols] * [len(home_before_cols)-i for i in range(len(home_before_cols))]).sum(axis=1)
    df['AwayForm'] = (df[away_before_cols] * [len(away_before_cols) - i for i in range(len(away_before_cols))]).sum(axis=1)
    return df
