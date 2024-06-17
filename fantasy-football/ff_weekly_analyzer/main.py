# %%
import pandas as pd
import numpy as np
from sleeper_wrapper import League

LEAGUE_ID = '991866561344454656'

_user_id_map = {
    '211184907353849856': 'Nithin',
    '468469793041674240': 'Jack',
    '473203024827772928': 'Anthony',
    '475034269295570944': 'Rishant',
    '730845846408687616': 'Zach',
    '730950227669958656': 'Drew',
    '730969117707853824': 'Ben',
    '730976912712794112': 'Nikhil',
    '731598867140837376': 'Stephen',
    '736402773066850304': 'Jashn',
    '991895710847102976': 'Ritwick',
    '870169428833284096': 'Venkat',
    '731691932765491200': 'Nishal'
}


def map_roster_ids(league) -> dict:
    _roster_id_map = {roster['roster_id']: _user_id_map[roster['owner_id']]
                      for roster in league.get_rosters()}
    return _roster_id_map


def weekly_matchups(league, week: int) -> list:
    keys = ['roster_id', 'points', 'matchup_id']
    matchups = [{**{key: team[key] for key in keys}, **{'week': week}}
                for team in league.get_matchups(week)]
    return matchups


def get_all_matchups(league, weeks_completed: int) -> list:
    all_matchups = []
    for week in range(1, weeks_completed+1):
        all_matchups.extend(weekly_matchups(league, week))
    return all_matchups


def get_matchup_df(all_matchups: list, n_teams: int) -> pd.DataFrame:
    df = pd.DataFrame(all_matchups)
    df['wins'] = df['win'] = np.where(df.groupby(
        ['matchup_id', 'week']).points.transform(max) == df.points, 1, 0)
    df['week_rank'] = (n_teams+1) - df.groupby('week').points.transform('rank')
    df['theoretical_wins'] = (n_teams-df.week_rank) / (n_teams-1)
    return df


def survivor_pool(matchup_df, n_teams: int, weeks_completed: int) -> list:
    roster_ids = list(range(1, n_teams+1))
    for week in range(1, weeks_completed+1):
        if len(roster_ids) > 1:
            rank_to_drop = matchup_df[(matchup_df.week == week) & (
                matchup_df.roster_id.isin(roster_ids))].week_rank.max()
            roster_id_to_drop = matchup_df[(matchup_df.week == week) & (
                matchup_df.week_rank == rank_to_drop)].roster_id.iloc[0]
            roster_ids.remove(roster_id_to_drop)

    users_remaining = [_roster_id_map[roster_id] for roster_id in roster_ids]
    return users_remaining


def wins_above_expected(matchup_df) -> dict:
    results = (matchup_df.groupby('roster_id').wins.sum() - matchup_df.groupby(
        'roster_id').theoretical_wins.sum()).rename('wins_above_expected').reset_index()
    results['user'] = results.roster_id.map(_roster_id_map)
    results.sort_values(by='wins_above_expected',
                        ascending=False, inplace=True)
    results.wins_above_expected = round(results.wins_above_expected, 2)
    results.set_index('user', inplace=True)
    return results['wins_above_expected'].to_dict()


def league_winnings(matchup_df, league_status: str, survivor_pool_users: list):

    current_winnings = pd.Series(
        index=_user_id_map.values()).rename('winnings').fillna(0)

    # survivor pool winner
    if len(survivor_pool_users) == 1:
        current_winnings[survivor_pool_users[0]] += 30

    # weekly high scorers
    high_scorers = matchup_df[matchup_df.week_rank == 1].groupby(
        'roster_id').wins.sum().reset_index()
    high_scorers['user'] = high_scorers.roster_id.map(_roster_id_map)
    high_scorers.set_index('user', inplace=True)
    high_scorers['earnings'] = high_scorers.wins*10

    for user, earnings in high_scorers['earnings'].to_dict().items():
        current_winnings[user] += earnings

    # league winners
    if league_status == 'complete':
        pass

    return current_winnings[current_winnings > 0].sort_values(ascending=False).astype(int)


league = League(LEAGUE_ID)
settings = league.get_league()
N_TEAMS = settings['total_rosters']
LEAGUE_STATUS = settings['status']
LEAGUE_NAME = settings['name']
WEEKS_COMPLETED = 2
_roster_id_map = map_roster_ids(league)
all_matchups = get_all_matchups(league, WEEKS_COMPLETED)

matchup_df = get_matchup_df(all_matchups, N_TEAMS)
survivor_pool_users = survivor_pool(matchup_df, N_TEAMS, WEEKS_COMPLETED)
wins_above_expected_results = wins_above_expected(matchup_df)
current_league_winnings = league_winnings(
    matchup_df, LEAGUE_STATUS, survivor_pool_users)
# %%


def result_string(survivor_pool_users, wins_above_expected_results, current_league_winnings, LEAGUE_NAME):
    wins_above_expected_string = ',\n'.join(
        [f'{name}: {wins}' for name, wins in wins_above_expected_results.items()])

    string = f"""
    Here is your weekly {LEAGUE_NAME} recap
    brought to you by Good Vibes Only:

    -------------------------
    Survivors of the Gulag
    -------------------------

    {wins_above_expected_string}

    -------------------------
    Whose Getting Lucky?
    -------------------------

    stuff 

    -------------------------
    Playoff Likelihoods
    -------------------------

    Coming after Week 3...
    Stay tuned for your
    destinies :) 

    -------------------------
    Current League Winnings
    -------------------------

    stuff


    """
# %%
