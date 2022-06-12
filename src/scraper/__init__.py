from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
from typing import List


class Scraper:
    def __init__(self, tm_url: str = "https://www.transfermarkt.co.uk"):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
        self.tm_url = tm_url

    def _get_page_soup(self, url_extension: str):
        url = self.tm_url + url_extension
        page = requests.get(url, headers={'User-Agent': self.user_agent})
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup

    def _get_match_information(self, match: BeautifulSoup):
        summary = match.find(class_='table-grosse-schrift')
        result_box = summary.find(class_="ergebnis-box")
        match_link: str = result_box.a['href']
        match_id = match_link[match_link.rfind('/') + 1:]
        match_result = result_box.find(class_="matchresult").text
        goals_home = int(match_result[:match_result.find(":")])
        goals_away = int(match_result[match_result.find(":") + 1:])
        result = "H" if goals_home > goals_away else "D" if goals_home == goals_away else "A"
        table_positions = summary.find_all(class_="tabellenplatz")
        tp_home = int(table_positions[0].text[1:-2])
        tp_away = int(table_positions[2].text[1:-2])

        teams = summary.find_all(class_="spieltagsansicht-vereinsname")
        home_team_link = teams[0].find_all("a")[1]['href']
        home_team_id = home_team_link[home_team_link.find("verein/") + 7: home_team_link.find("/saison_id")]
        away_team_link = teams[2].find_all("a")[0]['href']
        away_team_id = away_team_link[away_team_link.find("verein/") + 7: away_team_link.find("/saison_id")]

        datetime_info = match.find_all("tr")[1].find("td")
        date = datetime_info.find('a').text.replace("\n", "").strip()
        date_formatted = pd.to_datetime(date)
        time = datetime_info.contents[-1].replace('\n', '').replace('\xa0', '').replace('-', '').replace('\t',
                                                                                                         '').strip()

        match_meta = match.find_all("tr")[2].find("td")
        attendance = int(
            match_meta.contents[2].replace('\n', '').replace('\xa0', '').replace('·', '').replace('\t', '').replace('.',
                                                                                                                    '').strip())
        referee_link = match_meta.find('a')['href']
        referee_id = int(referee_link[referee_link.rfind("/") + 1:])

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
            print(action_info)
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


    def get_match_data(self, seasons: List[int], leagues: List[str], matchdays: List[int] = None):
        for league in leagues:
            for season in seasons:
                for matchday in matchdays:
                    url_extension = f"/wettbewerb/spieltag/wettbewerb/{league}/spieltag/{matchday}/saison_id/{season}"
                    soup = self._get_page_soup(url_extension)
                    all_tables = soup.find_all("tbody")
                    matches = all_tables[1:-3]
                    matches_information = [self._get_match_information(match) for match in matches]
                    matches_df = pd.DataFrame(matches_information)