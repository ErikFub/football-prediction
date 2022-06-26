from src.data.access_old import DbAccessLayer as DbAccessLayerOld, ExternalDataAccessLayer
from src.data.access import DbAccessLayer, RawDataAccessLayer
from unidecode import unidecode
import pandas as pd
from typing import Dict, List
from pandas.io import sql
import warnings
warnings.filterwarnings(action='ignore')


def run_team_name_matcher():
    teams_df = DbAccessLayer().load_table('Team')
    old_teams_df = DbAccessLayerOld().load_table('Team')

    db_access = DbAccessLayer()
    relations = db_access.load_table("TeamOdds")
    unrelated_teams_idx = [i for i, row in teams_df.iterrows() if row['team_id'] not in relations['team_id'].unique()]
    unrelated_teams = teams_df.loc[unrelated_teams_idx]

    print(f"{len(unrelated_teams_idx)} teams to go")

    for i, row in unrelated_teams.iterrows():
        tm_team = row['name']
        team_tokens = tm_team.split(" ")
        matches = []
        for token in team_tokens:
            token = unidecode(token).lower()
            for j, odds_row in old_teams_df.iterrows():
                odds_team = odds_row['Name']
                odds_team_tokens = [unidecode(e).lower() for e in odds_team.split(" ")]
                if token in odds_team_tokens and token != "fc":
                    matches.append(odds_row)
        if len(matches) > 0:
            matches_df = pd.concat([pd.DataFrame(match).transpose() for match in matches]).drop_duplicates().\
                reset_index()
            print(f"Matches for {tm_team}:")
            for k, match_row in matches_df.iterrows():
                print(f"\t{k+1} - {match_row['Name']}")
            user_in = input("Please insert the numbers you want to establish a connection for, separated by commas. If "
                            "there is no valid match, insert '-'.\n")
            if user_in == "-":
                continue
            else:
                idx_to_add = [int(e.strip())-1 for e in user_in.split(",")]
                for idx in idx_to_add:
                    odds_team_match = matches_df.loc[idx]
                    db_access.save_team_odds_relation(team_id=row['team_id'], odds_team_name=odds_team_match['Name'],
                                                      odds_team_id=odds_team_match['TeamID'])
        else:
            print(f"{tm_team} skipped")
        print("\n")


class Database:
    """
    Creation and update of database.
    """
    def __init__(self):
        self.db_access = DbAccessLayer()

    def create_matches_table(self):
        all_matches = RawDataAccessLayer().load_all_prepared()
        self.db_access.overwrite_table_from_df(all_matches, 'MatchSecondary')


class RawData:
    @staticmethod
    def prepare(country: str = 'all'):
        access_layer = RawDataAccessLayer()
        dirs_to_prepare = [country] if country != 'all' else [folder for folder in access_layer.subdirectories if folder != 'prepared']
        for country_dir in dirs_to_prepare:
            all_files: list = access_layer.get_subdir_files(country_dir)
            for file in all_files:
                df = access_layer.load_df(country_dir, file)
                df.dropna(how='all', inplace=True)
                dates = df['Date'].apply(lambda d: pd.to_datetime(d, format="%d/%m/%Y") if len(d) == 10 else pd.to_datetime(d, format="%d/%m/%y"))
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
                df['Date'] = df['Date'].apply(lambda d: pd.to_datetime(d, format="%d/%m/%Y") if len(d) == 10 else pd.to_datetime(d, format="%d/%m/%y"))
                df['Season'] = f"{min_year}/{max_year}"
                valid_cols = [col for col in df.columns if col[:7] != "Unnamed"]
                df_valid = df[valid_cols]
                df_valid.dropna(subset=['HomeTeam', 'AwayTeam', 'Div', 'Country'], inplace=True)
                access_layer.save_prepared(df_valid, new_file_name)
