import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import cachetools

def cached_function(ttl=3600, maxsize=100):
    """Decorator to create a unique cache for each function."""
    def decorator(func):
        cache = cachetools.TTLCache(maxsize=maxsize, ttl=ttl)  # Create a unique cache for each function
        
        @cachetools.cached(cache)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

teams = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Ipswich Town",
    "Leicester",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle",
    "Nottingham Forest",
    "Southampton",
    "Tottenham",
    "West Ham",
    "Wolves",
]


@cached_function(ttl=3600, maxsize=100)
def get_stats():
    url = "https://www.soccerstats.com/table.asp?league=england&tid=f"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, features="html.parser")
    scored_data_side = soup.find_all(id="btable")[1:3]
    scored_data_result = defaultdict(dict, defaultdict(dict))
    for i, side in enumerate(scored_data_side):
        home = True
        if i == 1:
            home = False
        scored_data_row = side.find_all("tr")[2:22]
        for j, row in enumerate(scored_data_row):
            (
                gf_4plus,
                gf_3,
                gf_2,
                gf_1,
                gf_0,
                gf_avg,
                team,
                ga_avg,
                ga_0,
                ga_1,
                ga_2,
                ga_3,
                ga_4plus,
            ) = row.find_all("td")
            goals_for = [
                (gf_4plus.text.strip()),
                (gf_3.text.strip()),
                (gf_2.text.strip()),
                (gf_1.text.strip()),
                (gf_0.text.strip()),
                (gf_avg.text.strip()),
            ]
            for i, stat in enumerate(goals_for):
                try:
                    goals_for[i] = float(stat)
                except:
                    goals_for[i] = 0
            goals_against = [
                (ga_4plus.text.strip()),
                (ga_3.text.strip()),
                (ga_2.text.strip()),
                (ga_1.text.strip()),
                (ga_0.text.strip()),
                (ga_avg.text.strip()),
            ]
            for i, stat in enumerate(goals_against):
                try:
                    goals_against[i] = float(stat)
                except:
                    goals_against[i] = 0
            if home:
                scored_data_result[teams[j]]["home"] = {
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                }
            else:
                scored_data_result[teams[j]]["away"] = {
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                }
    return scored_data_result
                

mapped_teams = {
    "Arsenal":"Arsenal",
    "Aston Villa":"Aston Villa",
    "Bournemouth":"Bournemouth",
    "Brentford":"Brentford",
    "Brighton":"Brighton",
    "Chelsea":"Chelsea",
    "Crystal Palace":"Crystal Palace",
    "Everton":"Everton",
    "Fulham":"Fulham",
    "Ipswich Town":"Ipswich Town",
    "Leicester City":"Leicester",
    "Liverpool":"Liverpool",
    "Manchester City":"Manchester City",
    "Manchester Utd":"Manchester United",
    "Nottm Forest":"Newcastle",
    "Newcastle Utd":"Nottingham Forest",
    "Southampton":"Southampton",
    "Tottenham":"Tottenham",
    "West Ham Utd": "West Ham",
    "Wolverhampton":"Wolves",
}
@cached_function(ttl=3600, maxsize=100)
def get_form():
    url = "https://www.soccerstats.com/formtable.asp?league=england"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, features="html.parser")
    form_table = soup.find_all(id="btable")[9].find_all("tr")[2:22]
    results = {}
    for form in form_table:
        team,_,_,_,form_of_team = form.find_all("td")[1:6]
        results[mapped_teams.get(team.text.strip())] = float(form_of_team.text.strip())
    return results

@cached_function(ttl=3600, maxsize=100)
def get_relative_performance():
    url = "https://www.soccerstats.com/formtable.asp?league=england"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, features="html.parser")
    form_table = soup.find_all(id="btable")[9].find_all("tr")[2:22]
    results = {}
    for form in form_table:
        team = form.find_all("td")[1]
        rp = form.find_all("td")[-2]
        results[mapped_teams.get(team.text.strip())] = float(rp.text.strip("%"))/100
    return results


@cached_function(ttl=3600, maxsize=100)
def get_top_scorers():
    # Custom team names
    custom_team_names = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea",
        "Crystal Palace", "Everton", "Fulham", "Ipswich Town", "Leicester", "Liverpool",
        "Manchester City", "Manchester United", "Newcastle", "Nottingham Forest",
        "Southampton", "Tottenham", "West Ham", "Wolves"
    ]

    # Fetch data from FPL API
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()

    # Extract teams and players data
    teams = data['teams']  # Teams data
    players = data['elements']  # Players data

    # Convert teams and players data to DataFrame for easier access
    teams_df = pd.DataFrame(teams)[['id', 'name']]  # Keep only team id and team name
    players_df = pd.DataFrame(players)[['first_name', 'second_name', 'team', 'goals_scored', 'threat']]

    # Create a mapping of team ID to custom team names (IDs are 1-indexed in FPL API)
    team_id_to_name = {i+1: custom_team_names[i] for i in range(len(custom_team_names))}

    # Add custom team names to the players DataFrame
    players_df['team_name'] = players_df['team'].map(team_id_to_name)

    # Select relevant columns for display (use 'team_name' instead of default team name)
    player_stats = players_df[['first_name', 'second_name', 'team_name', 'goals_scored', 'threat']]

    # Sort the players by team and goals scored
    player_stats = player_stats.sort_values(by=['team_name', 'goals_scored', 'threat'], ascending=[False, False, False])

    # Function to get top 3 scorers per team
    def _get_top_scorers(player_stats, top_n=3):
        top_scorers_per_team = player_stats.groupby('team_name').head(top_n)
        return top_scorers_per_team

    # Get the top 3 scorers for each team
    top_3_scorers = _get_top_scorers(player_stats)

    return top_3_scorers

@cached_function(ttl=3600, maxsize=100)
def get_top_booked():
    # Custom team names
    custom_team_names = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea",
        "Crystal Palace", "Everton", "Fulham", "Ipswich Town", "Leicester", "Liverpool",
        "Manchester City", "Manchester United", "Newcastle", "Nottingham Forest",
        "Southampton", "Tottenham", "West Ham", "Wolves"
    ]

    # Fetch data from FPL API
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()

    # Extract teams and players data
    teams = data['teams']  # Teams data
    players = data['elements']  # Players data

    # Convert teams and players data to DataFrame for easier access
    teams_df = pd.DataFrame(teams)[['id', 'name']]  # Keep only team id and team name
    players_df = pd.DataFrame(players)[['first_name', 'second_name', 'team', 'yellow_cards', 'minutes']]

    # Create a mapping of team ID to custom team names (IDs are 1-indexed in FPL API)
    team_id_to_name = {i+1: custom_team_names[i] for i in range(len(custom_team_names))}

    # Add custom team names to the players DataFrame
    players_df['team_name'] = players_df['team'].map(team_id_to_name)

    # Select relevant columns for display (use 'team_name' instead of default team name)
    player_stats = players_df[['first_name', 'second_name', 'team_name', 'yellow_cards', 'minutes']]

    # Sort the players by team and goals scored
    player_stats = player_stats.sort_values(by=['team_name', 'yellow_cards', 'minutes'], ascending=[True, False, False])

    # Function to get top 3 scorers per team
    def _get_top_booked(player_stats, top_n=3):
        top_scorers_per_team = player_stats.groupby('team_name').head(top_n)
        return top_scorers_per_team

    # Get the top 3 scorers for each team
    top_booked = _get_top_booked(player_stats)

    return top_booked

get_relative_performance()