import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_player_stats(year: int, player_type: str) -> pd.DataFrame:
    """
    Uses Pandas' built in HTML parser to scrape the tabular player statistics from
    https://www.hockey-reference.com/leagues/ . If the player played on multiple 
    teams in a single season, the individual team's statistics are discarded and
    the total ('TOT') statistics are retained (the multiple team names are discarded)

    Args:
        year (int): The first year of the season to retrieve, i.e. for the 2016-17
            season you'd put in 2016
        player_type (str): Either 'skaters' for forwards and defensemen, or 'goalies'
            for goaltenders.
    """

    if player_type not in ["skaters", "goalies"]:
        raise RuntimeError("'player_type' must be either 'skaters' or 'goalies'")

    url = f'https://www.hockey-reference.com/leagues/NHL_{year}_{player_type}.html'

    print(f"Retrieving data from '{url}'...")

    # Use Pandas' built in HTML parser to retrieve the tabular data from the web data
    # Uses BeautifulSoup4 in the background to do the heavylifting
    df = pd.read_html(url, header=1)[0]

    # get players which changed teams during a season
    players_multiple_teams = df[df['Tm'].isin(['TOT'])]

    # filter out players who played on multiple teams
    df = df[~df['Player'].isin(players_multiple_teams['Player'])]
    df = df[df['Player'] != "Player"]

    # add the aggregate rows
    df = df.append(players_multiple_teams, ignore_index=True)

    return df

def main():
    """
    Although, in general, the SV% is a good metric to analyse the performance of a goalie, using SV% by itself has it's own issues.

    - The main one being the number of games played by the goalies. For example, goalies who played one game vs another goalie who played 50 games. If a goalie played only one game and ended the game with zero goals conceded, he will have an average of 100.00 SV% for this metric. In contrast, a goalie who played 50 games and has an average of 95.00 SV% will look like he was worst than the one with 100.00 SV%. Where in reality, the goalie who played only one game might had an "easier game" or only had to save one shot, which leads to our second issue.

    - Another issue is the number of shots received by a goalie. The SV% alone does not differentiate between high and low number of shots received. Therefore, a goalie that received 10 shots in total and conceded 0 goals (100.00 SV%) will look better SV% wise than another goalie that received 1000 shots in total but only conceded 10 goals (99.00 SV%).

    - A better metric would be to combine the number of games played and the total number of shots received in order to filter the goalies by SV%.

    """
    # Get appropriate data from the nhl api
    goalies = get_player_stats(2017, "goalies")

    # Cast strings to appropriate numerical types
    goalies['SV%'] = goalies['SV%'].astype('float')
    goalies['GP'] = goalies['GP'].astype('int64')
    goalies['SA'] = goalies['SA'].astype('int64')

    # Drop rows with NAN value from SV% column
    goalies.dropna(subset=['SV%'], inplace=True)

    # Filter out goalies with less than 10 games played
    mask = goalies['GP'] >= 10
    goalies = goalies[mask]

    # print(goalies.sort_values(by='SV%', ascending=False).head(20))
    # print(goalies[["Player", "GP", "SV%", "SV"]].sort_values(by='SV%', ascending=False))
    # print(goalies.min())


    # Get top golies only
    goalies = goalies.sort_values(by='SV%', ascending=False).head(20)

    # Plot
    sns.set_theme(style="whitegrid")
    sns.barplot(x="SV%", y="Player", data=goalies, label="Total", color="b").set(
        title='Top 20 Goalies Save Percentage from 2017-2018 Season',
        xlabel='SV%', ylabel='Player Name')

    plt.show()


if __name__ == "__main__":
    main()