from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
from typing import List, Union, Dict
from src.data.access import DbAccessLayer
from time import sleep
from tqdm import tqdm


class Scraper:
    """Class for the scraping of relevant football data from transfermarkt.com. The locale can be specified by adapting
    the tm_url (e.g. https://www.transfermarkt.de for German locale)."""
    def __init__(self, tm_url: str = "https://www.transfermarkt.co.uk"):
        self._user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/" \
                           "102.0.0.0 Safari/537.36"
        self._tm_url = tm_url
        self._db_access = DbAccessLayer()

    def _get_page_soup(self, url_extension: str) -> BeautifulSoup:
        """Gets the BeautifulSoup content of the weppage specified through the defined URL extension."""
        url = self._tm_url + url_extension
        page = requests.get(url, headers={'User-Agent': self._user_agent})
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup

    @staticmethod
    def _get_match_information(match: BeautifulSoup):
        """Retrieves relevant information from transfermarkt.com matchday overview table entry."""
        summary = match.find(class_='table-grosse-schrift')
        result_box = summary.find(class_="ergebnis-box")
        match_result = result_box.find(class_="matchresult").text
        if match_result == 'postponed':
            match_link = match.parent.parent.find(title="Match report")['href']
            match_id = match_link[match_link.rfind('/') + 1:]
            # goals_home and goals_away will be set under section actions
        else:
            match_link: str = result_box.a['href']
            match_id = match_link[match_link.rfind('/') + 1:]
            goals_home = int(match_result[:match_result.find(":")])
            goals_away = int(match_result[match_result.find(":") + 1:])

        teams = summary.find_all(class_="spieltagsansicht-vereinsname")
        home_team_has_thread = len(teams[0].find_all("a")) > 1
        home_team_link = teams[0].find_all("a")[1 if home_team_has_thread else 0]['href']
        home_team_id = home_team_link[home_team_link.find("verein/") + 7: home_team_link.find("/saison_id")]
        away_team_link = teams[2].find_all("a")[0]['href']
        away_team_id = away_team_link[away_team_link.find("verein/") + 7: away_team_link.find("/saison_id")]

        table_positions = summary.find_all(class_="tabellenplatz")
        if len(table_positions) == 4:
            tp_home = int(table_positions[0].text[1:-2])
            tp_away = int(table_positions[2].text[1:-2])
        else:
            matchday_table = match.parent.parent.parent.parent.parent.find_all("table")[-2].find("tbody")
            table_rows = matchday_table.find_all("tr")
            for row in table_rows:
                team_link = row.find_all("td")[2].a['href']
                table_team_id = team_link[team_link.find("/verein/") + 8:  team_link.find("/saison_id")]
                prior_week_position = row.find_all("td")[0].span['title'][14:-6]
                if table_team_id == home_team_id:
                    tp_home = prior_week_position
                elif table_team_id == away_team_id:
                    tp_away = prior_week_position

        datetime_info = match.find_all("tr")[1].find("td")
        date = datetime_info.find('a').text.replace("\n", "").strip()
        date_formatted = pd.to_datetime(date)
        time = datetime_info.contents[-1].replace('\n', '').replace('\xa0', '').replace('-', '').replace('\t',
                                                                                                         '').strip()

        match_meta = match.find_all("tr")[2].find("td")
        has_attendance = len(match_meta.find_all(class_='icon-zuschauer-zahl')) > 0
        if has_attendance:
            attendance = int(match_meta.contents[2].replace('\n', '').replace('\xa0', '').replace('·', '').
                             replace('\t', '').replace('.', '').strip())
        else:
            attendance = np.NaN

        has_referee = len(match_meta.find_all(class_='icon-schiedsrichter')) > 0
        if has_referee:
            referee_link = match_meta.find('a')['href']
            referee_id = int(referee_link[referee_link.rfind("/") + 1:])
        else:
            referee_id = np.NaN

        all_actions = match.find_all(class_='spieltagsansicht-aktionen')
        goal_mins_home = []
        goal_mins_away = []
        goal_scorers_home = []
        goal_scorers_away = []
        for action in all_actions:
            cols = action.find_all("td")
            home_action: bool = cols[0].text != "\xa0" and cols[1].text != "\xa0"
            action_info = cols[0] if home_action else cols[4]
            has_icon = action_info.find("span", {'class': 'icons_sprite'}) is not None
            action_description = action_info.find("span", {'class': 'icons_sprite'}).get('title',
                                                                                         '').lower() if has_icon else ''
            is_goal = 'goal' in action_description
            if is_goal:
                minute = action_description[7: action_description.find(":")]
                scorer_link = action_info.find('a')['href']
                scorer_id = scorer_link[scorer_link.rfind("/") + 1:]
                if home_action:
                    goal_mins_home.append(minute)
                    goal_scorers_home.append(scorer_id)
                else:
                    goal_mins_away.append(minute)
                    goal_scorers_away.append(scorer_id)
        if match_result == 'postponed':
            goals_home = len(goal_mins_home)
            goals_away = len(goal_mins_away)
        result = "H" if goals_home > goals_away else "D" if goals_home == goals_away else "A"

        has_motm = len(match.find_all(class_="icon-spieler-des-spiels")) > 0
        if has_motm:
            motm_section = match.find(class_="icon-spieler-des-spiels").parent
            motm_link = motm_section.find('a')['href']
            motm_id = motm_link[motm_link.rfind("/") + 1:]
        else:
            motm_id = np.NaN
        return {
            'match_id': match_id,
            'date': date_formatted,
            'time': time,
            'attendance': attendance,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'table_position_home': tp_home,
            'table_position_away': tp_away,
            'goals_home': goals_home,
            'goals_away': goals_away,
            'result': result,
            'referee_id': referee_id,
            'goal_mins_home': ','.join(goal_mins_home),
            'goal_mins_away': ','.join(goal_mins_away),
            'goal_scorers_home': ','.join(goal_scorers_home),
            'goal_scorers_away': ','.join(goal_scorers_away),
            'motm_id': motm_id
        }

    def get_match_data(self, seasons: List[int], leagues: List[str], matchdays: Union[List[int], int], sleep_time: int = 5):
        """Scrapes and saves data for matches in specified leagues, seasons, and matchdays."""
        if isinstance(matchdays, int):
            matchdays = [i+1 for i in range(matchdays)]
        for league in leagues:
            for season in seasons:
                for matchday in matchdays:
                    url_extension = f"/wettbewerb/spieltag/wettbewerb/{league}/spieltag/{matchday}/saison_id/{season}"
                    soup = self._get_page_soup(url_extension)
                    all_tables = soup.find_all("tbody")
                    matches = [table for table in all_tables if
                               table.find_all("tr")[0].get('class', '') == ["table-grosse-schrift"]]
                    matches_information = [self._get_match_information(match) for match in matches]
                    matches_df = pd.DataFrame(matches_information)
                    matches_df['season'] = f"{str(season)[-2:]}/{str(int(str(season)[-2:])+1).rjust(2, '0')}"
                    matches_df['league_id'] = league
                    matches_df['matchday'] = matchday
                    self._db_access.save_matches(matches_df)
                    sleep(sleep_time)

    @staticmethod
    def _get_team_information(soup: BeautifulSoup) -> Dict:
        """Returns relevant elements on team information from html soup."""
        team_overview = soup.find("div", class_="dataMain")
        team_name = team_overview.find(itemprop="name").span.text

        team_details = soup.find("div", class_="dataContent")
        stadium_capacity = team_details.find(text="Stadium:").parent.parent.find(class_="tabellenplatz").text.replace(
            " Seats", "").replace(".", "")

        info_table = soup.find(class_="info-table")
        street = info_table.find(itemprop="streetAddress").text
        postal_code = info_table.find(itemprop="postalCode").text
        country = info_table.find(itemprop="addressLocality").text
        founded = pd.to_datetime(info_table.find(itemprop="foundingDate").text).strftime("%Y-%m-%d")
        return {
            'name': team_name,
            'founded': founded,
            'stadium_capacity': stadium_capacity,
            'street': street,
            'postal_code': postal_code,
            'country': country
        }

    def get_team_data(self, teams: List[int], sleep_time: int = 5):
        """Scrapes and saves data for all specified teams."""
        for team_id in tqdm(teams, desc="Getting team data"):
            soup = self._get_page_soup(f"/verein/startseite/verein/{team_id}/saison_id/2021")
            team_information = self._get_team_information(soup)
            team_information['team_id'] = team_id
            team_information_series = pd.Series(team_information)
            self._db_access.save_teams(team_information_series)
            sleep(sleep_time)

    def fill_missing_team_data(self, n_max: int = None):
        """Checks for which teams there exist matches but no dedicated entry in table 'Team' and saves the team data to
        the DB after scraping it."""
        matches_df = self._db_access.load_table("Match")
        teams_in_matches = list(set(matches_df['home_team_id'].tolist() + matches_df['away_team_id'].tolist()))
        existing_teams = self._db_access.load_table("Team")['team_id'].tolist()
        non_existent_teams = [team for team in teams_in_matches if team not in existing_teams]
        if n_max:
            self.get_team_data(non_existent_teams[:n_max])
        else:
            self.get_team_data(non_existent_teams)

    @staticmethod
    def _get_league_information(league_soup: BeautifulSoup) -> dict:
        """Returns relevant elements on league information from html soup."""
        name = league_soup.find("h1", class_="spielername-profil").text
        league_information = league_soup.find("div", class_="box-personeninfos")
        tier_information = league_information.find("th", text="League level:").parent.td
        tier_name_mapping = {'first': 1, 'second': 2, 'third': 3}
        tier_name = tier_information.text[:tier_information.text.find(" Tier")].replace("\n", "").strip().lower()
        tier = tier_name_mapping.get(tier_name, 0)
        country = tier_information.img['title']

        teams_information = league_information.find("th", text="Number of teams:").parent.td
        n_teams = int(teams_information.text[:teams_information.text.find(" teams")].replace("\n", "").strip())
        return {
            'name': name,
            'tier': tier,
            'n_teams': n_teams,
            'country': country
        }

    def get_league_data(self, leagues: List[str], sleep_time: int = 5):
        """Scrapes and saves data for all specified teams."""
        for league_id in tqdm(leagues, desc="Getting league data"):
            league_soup = self._get_page_soup(f"/wettbewerb/startseite/wettbewerb/{league_id}")
            league_information = self._get_league_information(league_soup)
            league_information['league_id'] = league_id
            league_information_series = pd.Series(league_information)
            self._db_access.save_leagues(league_information_series)
            sleep(sleep_time)

    def fill_missing_league_data(self):
        """Checks for which leagues there exist matches but no dedicated entry in table 'Team' and saves the team data
        to the DB after scraping it."""
        matches_df = self._db_access.load_table("Match")
        leagues_in_matches = matches_df['league_id'].unique().tolist()
        existing_leagues = self._db_access.load_table("League")['league_id'].tolist()
        non_existent_leagues = [league for league in leagues_in_matches if league not in existing_leagues]
        self.get_league_data(non_existent_leagues)


