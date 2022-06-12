import pandas as pd
from sqlalchemy import create_engine
import os
from typing import List


def get_main_dir_path() -> str:
    exec_directory = os.getcwd()
    main_directory = "football-prediction"
    main_dir_path = exec_directory[:exec_directory.find(main_directory) + len(main_directory)]
    return main_dir_path


class DbAccessLayer:
    def __init__(self):
        self._conn = create_engine(f"sqlite:///{get_main_dir_path()}//data//database_old")

    def create_table_from_df(self, df: pd.DataFrame, table_name: str) -> None:
        df.to_sql(table_name, con=self._conn, if_exists="fail", index=False)

    def append_to_table_from_df(self, df: pd.DataFrame, table_name: str) -> None:
        df.to_sql(table_name, con=self._conn, if_exists="append", index=False)

    def overwrite_table_from_df(self, df: pd.DataFrame, table_name: str) -> None:
        df.to_sql(table_name, con=self._conn, if_exists="replace", index=False)

    def load_table(self, table_name: str) -> pd.DataFrame:
        df = pd.read_sql(table_name, con=self._conn)
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


class ExternalDataAccessLayer:
    def __init__(self):
        self.dir = f"{get_main_dir_path()}/data/external"

    def get_country_data(self) -> pd.DataFrame:
        df = pd.read_csv(f"{self.dir}/all_countries.csv")
        return df
