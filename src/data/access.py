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

    def update_col_val(self, table: str, col: str, condition_col: str, condition_val: str, new_val):
        """Update the value of a column und a given condition."""
        sql.execute(f"""
        UPDATE {table} SET {col} = {"'" + str(new_val) + "'" if not pd.isna(new_val) else 'NULL'} 
        WHERE {condition_col} = '{condition_val}'""", con=self._conn)

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

    def update_feature(self, feature_vals: pd.Series):
        """Update vales in features table. Index of series must be match id and the name must correspond to the feature
         name."""
        for match_id, val in feature_vals.iteritems():
            self.update_col_val(table="Features", col=feature_vals.name, condition_col='match_id',
                                condition_val=match_id, new_val=val)

    def get_features_labels(self):
        features_df = self.load_table('Features')
        match_df = self.load_table('Match')
        labels = features_df.merge(match_df, on='match_id', how='left')[['id', 'result']]
        features = features_df.drop(columns=['match_id'])
        if len(labels) != len(features):
            raise Exception("Internal error: features and labels do not have the same length.")
        labels.index = match_df['date']
        features.index = match_df['date']
        return features, labels


class ExternalDataAccessLayer:
    """Class to access and save external data saved in/to folder data/external."""
    def __init__(self):
        self.dir = f"{get_main_dir_path()}/data/external"

    def get_country_data(self) -> pd.DataFrame:
        df = pd.read_csv(f"{self.dir}/all_countries.csv")
        return df


class RawDataAccessLayer:
    def __init__(self):
        self.dir = f"{get_main_dir_path()}/data/raw"

    @property
    def subdirectories(self) -> list:
        return os.listdir(self.dir)

    def get_subdir_files(self, subdir: str) -> List[str]:
        return os.listdir(f"{self.dir}/{subdir}")

    def load_df(self, subdir: str, file_name: str) -> pd.DataFrame:
        return pd.read_csv(f"{self.dir}/{subdir}/{file_name}", on_bad_lines='skip', encoding='ISO-8859-1')

    def save_prepared(self, df: pd.DataFrame, file_name: str) -> None:
        df.to_csv(f"{self.dir}/prepared/{file_name}", index=False)

    def load_all_prepared(self) -> pd.DataFrame:
        all_files = [file for file in os.listdir(f"{self.dir}/prepared") if file[-4:] == ".csv"]
        all_prepared = pd.DataFrame()
        for file in all_files:
            file_path = f"{self.dir}/prepared/{file}"
            df = pd.read_csv(file_path)
            all_prepared = pd.concat([all_prepared, df])
        return all_prepared
