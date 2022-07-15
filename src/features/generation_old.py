import pandas as pd
import numpy as np
from tqdm import tqdm
from src.data.access import DbAccessLayer
from typing import List, Dict, Tuple, Union


class FeaturesLabelsGenerator:
    def __init__(self):
        self.db_access = DbAccessLayer()
        self.match_df = self.db_access.load_table('Match')
        self.team_df = None
        self.odds_df = None
        self.features_df = pd.DataFrame(index=self.match_df.index)

    def _check_requirements(self, required: List[str]):
        allowed_requirements = ['match', 'odds', 'team']
        required = [r.lower() for r in required]
        for requirement in required:
            if requirement not in allowed_requirements:
                raise ValueError(f"Requirement '{requirement}' not supported. Please choose one of the following: "
                                 f"{', '.join(allowed_requirements)}")
            if requirement == 'match' and self.match_df is None:
                self.match_df = self.db_access.load_table('Match')
            if requirement == 'odds' and self.odds_df is None:
                self.odds_df = self.db_access.load_table('Odds')
            if requirement == 'team' and self.team_df is None:
                self.team_df = self.db_access.load_table('Team')

    def get_before_outcomes(self, n_before: int):
        """Adds the n outcomes of the prior matches of the home and away team."""
        before_cols = [f'home_before_{i+1}' for i in range(n_before)] + [f'away_before_{i+1}' for i in range(n_before)]

        # Create columns and set to NaN
        for col in before_cols:
            self.features_df[col] = np.NaN

        def fill_before_outcomes(team_id: int):
            """Fills the before outcomes of a given team."""
            team_match_df = self.match_df[(self.match_df['home_team_id'] == team_id) |
                                          (self.match_df['away_team_id'] == team_id)]
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
                            self.features_df.loc[i, col] = row[col[5:]]

        all_teams = set(self.match_df['home_team_id'].unique().tolist() + self.match_df['away_team_id'].unique().tolist())
        for team in tqdm(all_teams, desc="Adding before outcomes"):  # for each team, fill the outcomes
            fill_before_outcomes(team)

    def get_form(self, n_matches):
        """Add the form as a weighted average of the before outcomes where recent outcomes have more weight."""
        home_before_cols = [f'home_before_{i + 1}' for i in range(n_matches)]
        away_before_cols = [f'away_before_{i + 1}' for i in range(n_matches)]
        required_before_cols = home_before_cols + away_before_cols
        for before_col in required_before_cols:
            if before_col not in self.features_df.columns:
                self.get_before_outcomes(n_matches)
                break

        self.features_df['home_form'] = (self.features_df[home_before_cols] *
                                         [len(home_before_cols) - i for i in range(len(home_before_cols))]).sum(axis=1)
        self.features_df['away_form'] = (self.features_df[away_before_cols] *
                                         [len(away_before_cols) - i for i in range(len(away_before_cols))]).sum(axis=1)

    def get_bookmaker_pred(self):
        """Gets the prediction the bookmaker would have made. Configured for Bet365 as bookmaker."""
        self._check_requirements(['match', 'odds'])
        bookmaker_pred = self.odds_df.set_index('match_id')[['b365_H', 'b365_D', 'b365_A']].idxmin(axis=1).str[-1:]
        bookmaker_pred.name = 'bookmaker_pred'
        self.features_df = self.features_df.join(bookmaker_pred, how='left')

    def get_outcome_counts(self):
        """Counts the number of wins, draws, and losses per home team in the prior n matches where n is the number of
        columns that capture prior results (these columns must already exist in DataFrame)."""
        df = self.match_df.copy()
        sites = ['home', 'away']
        for site in sites:
            before_cols = [col for col in df.columns if col[:11] == f'{site}_before']
            wins = (df[before_cols] == 1).sum(axis=1)
            draws = (df[before_cols] == 0).sum(axis=1)
            losses = (df[before_cols] == -1).sum(axis=1)
            self.features_df[f"{site}_wins"] = wins
            self.features_df[f"{site}_draws"] = draws
            self.features_df[f"{site}_losses"] = losses

    def get_relative_attendance(self):
        self._check_requirements(['team'])
        df = self.match_df.merge(self.team_df, left_on='home_team_id', right_on='team_id', how='left')
        rel_attendance = df['attendance'].fillna(0) / df['stadium_capacity']
        self.features_df['rel_attendance'] = rel_attendance

    def get_attendance_coef(self):
        if 'rel_attendance' not in self.features_df.columns:
            self.get_relative_attendance()
        self.features_df['attendance_coef'] = self.features_df['rel_attendance'] * self.match_df['attendance'].fillna(0)

    def get_features_labels(self):
        self._check_requirements(required=['match'])
        self.get_form(10)
        self.get_bookmaker_pred()
        self.get_relative_attendance()
        self.get_attendance_coef()
        return self.features_df, self.match_df['result']
