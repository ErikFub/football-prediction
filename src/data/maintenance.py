import pandas as pd
from typing import Dict, List
from src.data.access import RawDataAccessLayer, ExternalDataAccessLayer, DbAccessLayer
from sqlalchemy import create_engine
from pandas.io import sql
import warnings
warnings.filterwarnings(action='once')


class Database:
    """
    Creation and update of database.
    """
    def __init__(self):
        self.db_access = DbAccessLayer()
        self._conn = self.db_access._conn
        self.country_df = None
        self.all_data_df = None
        self.team_df = None
        self.league_df = None

    def _check_prerequisites(self, required: List[str]):
        allowed = ['team', 'country', 'all_data', 'league']
        required = [e.lower() for e in required]
        for e in required:
            if e not in allowed:
                raise ValueError(f"{e} not supported. Choose one of {', '.join(allowed)}.")
            if e == 'all_data':
                if self.all_data_df is None:
                    self.all_data_df = RawDataAccessLayer().load_all_prepared()
            if e == 'team':
                if self.team_df is None:
                    table_exists = self._check_table_existence('Team')
                    if not table_exists:
                        self.create_teams_table()
                    self.team_df = self.db_access.load_table('Team')
            if e == 'country':
                if self.country_df is None:
                    table_exists = self._check_table_existence('Country')
                    if not table_exists:
                        self.create_country_table()
                    self.country_df = self.db_access.load_table('Country')
            if e == 'league':
                if self.league_df is None:
                    table_exists = self._check_table_existence('League')
                    if not table_exists:
                        self.create_leagues_table()
                    self.league_df = self.db_access.load_table('League')

    def _check_table_existence(self, table_name: str) -> bool:
        tables = pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';", self._conn)
        table_exists = len(tables) > 0
        return table_exists

    def create_country_table(self):
        # Delete existent table
        sql.execute('DROP TABLE IF EXISTS Country', self._conn)

        # Create new table
        sql.execute("""
        CREATE TABLE Country (CountryID INTEGER PRIMARY KEY,
        Name TEXT NOT NULL,
        IsoCode TEXT,
        Region TEXT,
        SubRegion TEXT);
        """, self._conn)

        country_df = ExternalDataAccessLayer().get_country_data()
        relevant_cols: Dict[str, str] = {
            'name': 'Name',
            'alpha-2': 'IsoCode',
            'region': 'Region',
            'sub-region': 'SubRegion'
        }
        country_df_relevant = country_df[relevant_cols.keys()].rename(columns=relevant_cols)
        country_df_relevant['CountryID'] = (country_df_relevant.index + 1).astype(int)
        self.db_access.append_to_table_from_df(country_df_relevant, "Country")

    def create_teams_table(self):
        # Delete existent table
        sql.execute('DROP TABLE IF EXISTS Team', self._conn)

        # Create new table
        sql.execute("""
                CREATE TABLE Team (TeamID INTEGER PRIMARY KEY,
                Name TEXT NOT NULL,
                CountryID INTEGER NOT NULL);
                """, self._conn)

        self._check_prerequisites(['all_data', 'country'])

        home_teams = self.all_data_df[['HomeTeam', 'Country']].rename(columns={'HomeTeam': 'Name'})
        away_teams = self.all_data_df[['AwayTeam', 'Country']].rename(columns={'AwayTeam': 'Name'})
        all_teams = pd.concat([home_teams, away_teams]).drop_duplicates()
        all_teams = all_teams.merge(self.country_df[['IsoCode', 'CountryID']], how='left', left_on='Country',
                                    right_on='IsoCode')
        all_teams.drop(columns=['Country', 'IsoCode'], inplace=True)
        all_teams['Name'].fillna('Unknown', inplace=True)
        self.db_access.append_to_table_from_df(all_teams, 'Team')

    def create_leagues_table(self):
        # Delete existent table
        sql.execute('DROP TABLE IF EXISTS League', self._conn)

        # Create new table
        sql.execute(""" CREATE TABLE League (LeagueID INTEGER PRIMARY KEY,
                        Name TEXT,
                        Division INTEGER NOT NULL,
                        CountryID INTEGER NOT NULL);
                        """, self._conn)

        self._check_prerequisites(['all_data', 'country'])

        all_leagues = self.all_data_df[['Div', 'Country']].drop_duplicates()
        all_leagues = all_leagues.merge(self.country_df[['IsoCode', 'CountryID']], how='left',
                                                     left_on='Country', right_on='IsoCode').drop(columns=['IsoCode'])
        all_leagues.drop(columns=['Country'], inplace=True)
        all_leagues.rename(columns={'Div': 'Division'}, inplace=True)
        all_leagues['Division'] = all_leagues['Division'].apply(lambda d: int(''.join([c for c in d if c.isnumeric()])))
        self.db_access.append_to_table_from_df(all_leagues, 'League')

    def create_matches_table(self):
        self._check_prerequisites(['all_data', 'country', 'league', 'team'])

        all_matches = self.all_data_df.copy()
        all_matches = all_matches.merge(self.country_df[['IsoCode', 'CountryID']], how='left', left_on='Country',
                                        right_on='IsoCode').drop(columns=['IsoCode'])

        all_matches['Div'] = all_matches['Div'].apply(lambda d: int(''.join([c for c in d if c.isnumeric()])))
        league = all_matches.merge(self.league_df, how='left', left_on=['Div', 'CountryID'],
                                   right_on=['Division', 'CountryID'])['LeagueID']
        all_matches.insert(0, 'LeagueID', league)
        all_matches.drop(columns=['Div'], inplace=True)

        home_team = all_matches.merge(self.team_df, how='left', left_on=['HomeTeam', 'CountryID'],
                                      right_on=['Name', 'CountryID'])['TeamID']
        all_matches.insert(2, 'HomeTeamID', home_team)
        all_matches.drop(columns=['HomeTeam'], inplace=True)

        away_team = all_matches.merge(self.team_df, how='left', left_on=['AwayTeam', 'CountryID'],
                                      right_on=['Name', 'CountryID'])['TeamID']
        all_matches.insert(3, 'AwayTeamID', away_team)
        all_matches.drop(columns=['AwayTeam'], inplace=True)

        all_matches.drop(columns=['Country', 'CountryID'], inplace=True)

        all_matches.insert(0, 'MatchID', all_matches.reset_index().index + 1)
        self.db_access.overwrite_table_from_df(all_matches, 'Match')

    def create(self):
        self.create_country_table()
        self.create_teams_table()
        self.create_leagues_table()
        self.create_matches_table()


class RawData:
    @staticmethod
    def prepare(country: str = 'all'):
        access_layer = RawDataAccessLayer()
        dirs_to_prepare = [country] if country != 'all' else [folder for folder in access_layer.subdirectories if folder != 'prepared']
        for country_dir in dirs_to_prepare:
            all_files: list = access_layer.get_subdir_files(country_dir)
            for file in all_files:
                df = access_layer.load_df(country_dir, file)
                dates = pd.to_datetime(df['Date'], infer_datetime_format=True)
                min_year = int(dates.dt.year.min())
                max_year = int(dates.dt.year.max())
                div: str = df['Div'].values[0]
                if country_dir.upper() == "EN":
                    div_num = int(''.join([e for e in div if e.isnumeric()])) + 1
                    div = ''.join([e for e in div if e.isalpha()]) + str(div_num)
                    df['Div'] = div
                new_file_name = f"{country_dir.upper()}_{div}_{str(int(min_year))[-2:]}-{str(int(max_year))[-2:]}.csv"
                df['Country'] = country_dir.upper()
                df['HomeTeam'] = df['HomeTeam'].apply(lambda n: str(n).strip())
                df['AwayTeam'] = df['AwayTeam'].apply(lambda n: str(n).strip())
                df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
                df['Season'] = f"{min_year}/{max_year}"
                valid_cols = [col for col in df.columns if col[:7] != "Unnamed"]
                df_valid = df[valid_cols]
                df_valid.dropna(subset=['HomeTeam', 'AwayTeam', 'Div', 'Country'], inplace=True)
                access_layer.save_prepared(df_valid, new_file_name)
