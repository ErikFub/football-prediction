import pandas as pd
from typing import Dict
from src.data.access import RawDataAccessLayer, ExternalDataAccessLayer, DbAccessLayer
from sqlalchemy import create_engine
from pandas.io import sql


class Database:
    def __init__(self):
        self.db_access = DbAccessLayer()
        self._conn = create_engine("sqlite:///C://Users//erikf//Desktop//football-prediction//data//database")

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
        country_df_relevant['CountryID'] = country_df_relevant.index + 1
        self.db_access.append_to_table_from_df(country_df_relevant, "Country")

    def create(self):
        self.create_country_table()


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
                min_year = dates.dt.year.min()
                max_year = dates.dt.year.max()
                div = df['Div'].values[0]
                new_file_name = f"{country_dir.upper()}_{div}_{str(int(min_year))[-2:]}-{str(int(max_year))[-2:]}.csv"
                df['Country'] = country_dir.upper()
                valid_cols = [col for col in df.columns if col[:7] != "Unnamed"]
                df_valid = df[valid_cols]
                access_layer.save_prepared(df_valid, new_file_name)
