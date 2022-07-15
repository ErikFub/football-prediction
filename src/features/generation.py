import pandas as pd
import numpy as np
from tqdm import tqdm
from src.data.access import DbAccessLayer
from typing import List, Dict, Tuple, Union
import warnings

warnings.filterwarnings('ignore')


class FeaturesLabelsGenerator:
    def __init__(self):
        self.db_access = DbAccessLayer()
        self.match_df = self.db_access.load_table('Match')
        self.team_df = None
        self.odds_df = None

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

    def get_before_outcomes(self, n_before: int, site_specific: bool = False):
        """Gets the n outcomes of the prior matches of the home and away team."""
        before_cols = [f'home_before_{i + 1}' for i in range(n_before)] + [f'away_before_{i + 1}' for i in
                                                                           range(n_before)]
        before_outcomes_df = pd.DataFrame(index=self.match_df['match_id'])

        # Create columns and set to NaN
        for col in before_cols:
            before_outcomes_df[col] = np.NaN

        def fill_before_outcomes(team_id: int, site: str = 'any'):
            """Fills the before outcomes of a given team."""
            if site not in ['any', 'home', 'away']:
                raise ValueError(f"Site '{site}' not supported.")
            # Get all matches of team
            if site == 'any':
                team_match_df = self.match_df[(self.match_df['home_team_id'] == team_id) |
                                              (self.match_df['away_team_id'] == team_id)]
            elif site == 'home':
                team_match_df = self.match_df[self.match_df['home_team_id'] == team_id]
            else:  # covers site == 'away'
                team_match_df = self.match_df[self.match_df['away_team_id'] == team_id]

            if len(team_match_df) > 0:
                team_match_df['team_role'] = team_match_df['home_team_id'].apply(
                    lambda ht: 'H' if ht == team_id else 'A')
                team_match_df['outcome'] = team_match_df.apply(
                    lambda r: 1 if r['result'] == r['team_role'] else 0 if r['result'] == 'D' else -1, axis=1)
                for i in range(n_before):
                    n = i + 1
                    team_match_df[f'before_{n}'] = team_match_df['outcome'].shift(n)

                for i, row in team_match_df.iterrows():
                    for col in before_cols:
                        row_role = 'home' if row['team_role'] == 'H' else 'away'
                        col_role = col[:4]
                        if row_role == col_role:
                            before_outcomes_df.loc[row['match_id'], col] = row[col[5:]]

        all_teams = set(
            self.match_df['home_team_id'].unique().tolist() + self.match_df['away_team_id'].unique().tolist())
        for team in tqdm(all_teams, desc="Adding before outcomes"):  # for each team, fill the outcomes
            if not site_specific:
                fill_before_outcomes(team)
            else:
                fill_before_outcomes(team, site='home')
                fill_before_outcomes(team, site='away')

        return before_outcomes_df

    def get_form(self, n_matches, site_specific: bool = False, weighted: bool = False):
        """Add the form as a weighted average of the before outcomes where recent outcomes have more weight."""
        home_before_cols = [f'home_before_{i + 1}' for i in range(n_matches)]
        away_before_cols = [f'away_before_{i + 1}' for i in range(n_matches)]
        before_outcomes_df = self.get_before_outcomes(n_matches, site_specific)

        feature_name_prefix = f"{'site_' if site_specific else ''}form_{'w_' if weighted else ''}"

        home_form = before_outcomes_df[home_before_cols].sum(axis=1) if not weighted \
            else (before_outcomes_df[home_before_cols] *
                  [len(home_before_cols) - i for i in range(len(home_before_cols))]).sum(axis=1)
        home_form.name = feature_name_prefix + 'home'

        away_form = before_outcomes_df[away_before_cols].sum(axis=1) if not weighted \
            else (before_outcomes_df[away_before_cols] *
                  [len(away_before_cols) - i for i in range(len(away_before_cols))]).sum(axis=1)
        away_form.name = feature_name_prefix + 'away'
        self.db_access.update_feature(home_form)
        self.db_access.update_feature(away_form)

    def get_bookmaker_pred(self):
        """Gets the prediction the bookmaker would have made. Configured for Bet365 as bookmaker."""
        self._check_requirements(['match', 'odds'])
        bookmaker_pred = self.odds_df.set_index('match_id')[['b365_H', 'b365_D', 'b365_A']].idxmin(axis=1).str[-1:]
        bookmaker_pred.name = 'bookmaker_pred'
        self.db_access.update_feature(bookmaker_pred)

    def get_goal_form(self, n_matches: int = 5):
        """Compute form of a team solely based on number of goals scored vs. conceded in the prior n matches."""
        self._check_requirements(['match'])
        all_teams = list(set(self.match_df['home_team_id'].tolist() + self.match_df['away_team_id'].tolist()))
        goals_form_home = pd.Series(index=self.match_df['match_id'], name='goals_form_home')
        goals_form_away = pd.Series(index=self.match_df['match_id'], name='goals_form_away')
        for team_id in tqdm(all_teams):
            team_match_df = self.match_df[(self.match_df['home_team_id'] == team_id) |
                                          (self.match_df['away_team_id'] == team_id)]
            team_match_df['team_role'] = team_match_df.apply(
                lambda r: 'home' if r['home_team_id'] == team_id else 'away', axis=1)
            team_match_df['match_goal_form'] = team_match_df.apply(
                lambda r: r['goals_home'] - r['goals_away'] if r['team_role'] == 'home'
                else r['goals_away'] - r['goals_home'], axis=1)
            team_match_df['goal_form'] = team_match_df['match_goal_form'].shift(1).rolling(n_matches, min_periods=1).sum()

            for i, row in team_match_df.iterrows():
                match_id = row['match_id']
                if row['team_role'] == 'home':
                    goals_form_home.loc[match_id] = row['goal_form']
                else:
                    goals_form_away.loc[match_id] = row['goal_form']

        self.db_access.update_feature(goals_form_home)
        self.db_access.update_feature(goals_form_away)

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
