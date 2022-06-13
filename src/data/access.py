import pandas as pd
from sqlalchemy import create_engine
import os
from typing import List, Union, Dict
from pandas.io import sql


def get_main_dir_path() -> str:
    """Gets the path to the main directory of this project. Requires the main directory to be named
    'football-prediction'."""
    exec_directory = os.getcwd()
    main_directory = "football-prediction"
    main_dir_path = exec_directory[:exec_directory.find(main_directory) + len(main_directory)]
    return main_dir_path


class DbAccessLayer:
    def __init__(self):
        self._conn = create_engine(f"sqlite:///{get_main_dir_path()}//data//database.db")

    def create_table_from_df(self, df: pd.DataFrame, table_name: str) -> None:
        """Create table from dataframe. If the table already exists, raise error."""
        df.to_sql(table_name, con=self._conn, if_exists="fail", index=False)

    def append_to_table_from_df(self, df: pd.DataFrame, table_name: str) -> None:
        """Appends to table from dataframe. If the table does not already exist, creates the table."""
        df.to_sql(table_name, con=self._conn, if_exists="append", index=False)

    def overwrite_table_from_df(self, df: pd.DataFrame, table_name: str) -> None:
        """Overwrites existing table with dataframe. If the table does not already exist, creates the table."""
        df.to_sql(table_name, con=self._conn, if_exists="replace", index=False)

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Loads table with given name."""
        df = pd.read_sql(table_name, con=self._conn)
        return df

    def _update_insert_entry(self, entry: pd.Series, table: str):
        """Updates if exists, or inserts an entry into a specified table of the database. All values in index of entry
        will be attempted to save to the table."""
        cols = entry.index.tolist()

        # Create the 'VALUES' string required in SQL statement
        values = ""
        for value in entry:
            if value is None or pd.isna(value):
                values += "NULL, "
            else:
                value = str(value).replace("'", "")
                values += f"'{value}', "
        values = values[:-2]  # remove trailing ", "

        sql.execute(f"""
            INSERT OR REPLACE INTO {table} ({", ".join(cols)})
            VALUES ({values});
        """, self._conn)

    def save_matches(self, matches: Union[pd.Series, pd.DataFrame]):
        if isinstance(matches, pd.Series):
            self._update_insert_entry(matches, 'Match')
        elif isinstance(matches, pd.DataFrame):
            for idx, row in matches.iterrows():
                self._update_insert_entry(row, 'Match')

    def save_teams(self, teams: Union[pd.Series, pd.DataFrame]):
        if isinstance(teams, pd.Series):
            self._update_insert_entry(teams, 'Team')
        elif isinstance(teams, pd.DataFrame):
            for idx, row in teams.iterrows():
                self._update_insert_entry(row, 'Team')

    def save_leagues(self, leagues: Union[pd.Series, pd.DataFrame]):
        if isinstance(leagues, pd.Series):
            self._update_insert_entry(leagues, 'League')
        elif isinstance(leagues, pd.DataFrame):
            for idx, row in leagues.iterrows():
                self._update_insert_entry(row, 'League')

    def save_team_odds_relation(self, team_id, odds_team_name, odds_team_id):
        relation = pd.Series(index=['team_id', 'odds_team_name', 'odds_team_id'], data=[team_id, odds_team_name,
                                                                                        odds_team_id])
        self._update_insert_entry(relation, "TeamOdds")

    def save_odds(self, odds_df: pd.DataFrame):
        for i, row in odds_df.iterrows():
            db_entry = pd.Series()
            db_entry['match_id'] = row['match_id']
            db_entry['b365_H'] = row['B365H']
            db_entry['b365_D'] = row['B365D']
            db_entry['b365_A'] = row['B365A']
            self._update_insert_entry(db_entry, 'Odds')


class ExternalDataAccessLayer:
    """Class to access and save external data saved in/to folder data/external."""
    def __init__(self):
        self.dir = f"{get_main_dir_path()}/data/external"

    def get_country_data(self) -> pd.DataFrame:
        df = pd.read_csv(f"{self.dir}/all_countries.csv")
        return df
