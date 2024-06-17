import pandas as pd
import numpy as np
from scipy.stats import norm, skewnorm
import itertools
from scipy import stats
import ast
from dataclasses import dataclass, field
import warnings

_POSITIONS = ['QB', 'RB', 'WR', 'TE']
_SIMULATIONS = 20000


@dataclass
class Settings:
    n_teams: int
    n_starters: dict
    third_round_reversal: bool
    roster_size: int
    adp_source: str = 'sleeper'
    use_injury_risk: bool = False
        
        
@dataclass
class UserTeam:
    settings: Settings
    first_pick: int
    players_drafted: list = field(init=False)
    starters_needed: dict = field(init=False)
    draft_picks: list = field(init=False)


    def __post_init__(self): 
        self.draft_picks = [self.draft_pick(round_for_pick) for round_for_pick in range(1,self.settings.roster_size+1)]
        self.players_drafted = []
        self.starters_needed = self.settings.n_starters


    def draft_pick(self, round_for_pick:int):
        if not self.settings.third_round_reversal or round_for_pick<3:
            return self.settings.n_teams*(round_for_pick-1)+(self.first_pick if round_for_pick%2==1 else self.settings.n_teams-self.first_pick+1)
        else:
            return self.settings.n_teams*(round_for_pick-1)+(self.first_pick if round_for_pick%2==0 else self.settings.n_teams-self.first_pick+1)
        
    
    def log_draft_pick(self, player: str, position: str):
        self.players_drafted.append({'player': player, 'position': position})

        if self.starters_needed[position]>0:
            self.starters_needed[position] -=1
        elif (position in ['RB', 'WR']) and (self.starters_needed['Flex']>0):
            self.starters_needed['Flex'] -=1

    def undo_user_pick(self):
        position = self.players_drafted[-1]['position']
        del self.players_drafted[-1]

        for position in _POSITIONS:
            if (position in ['RB', 'WR']) and (self.starters_needed['Flex']<0):
                self.starters_needed['Flex'] +=1
            else:
                self.starters_needed[position] +=1

        
@dataclass
class DraftState:
    all_players_df: pd.DataFrame
    settings: Settings
    user_team: UserTeam
    current_pick:int = 1
    current_round: int = 1
    drafted_players: list = field(init=False) 


    def __post_init__(self): 
        self.all_players_df.params = self.all_players_df.params.apply(ast.literal_eval)
        self.drafted_players=[]
        self.my_players = {}
        self.starters_needed = self.settings.n_starters

    
    def update_draft_state(self, drafted_players_from_app: list):

        if self.current_pick in self.user_team.draft_picks:
            player = drafted_players_from_app[-1]
            position = self.all_players_df[self.all_players_df.player==player].position.to_list()[0]
            self.user_team.log_draft_pick(player, position)

        if self.current_pick%self.settings.n_teams==0:
            self.current_round += 1
        self.current_pick +=1
        self.drafted_players = drafted_players_from_app
            

    def undo_last_pick(self):
        self.current_pick -=1
        if self.current_pick%self.settings.n_teams==0:
            self.current_round -=1
        del self.drafted_players[-1]

        if self.current_pick in self.user_team.draft_picks:
            self.user_team.undo_user_pick()
        

def player_simulation(players:pd.DataFrame, confidence_level_cap: float = 0.8):
    a_arr = np.array(players.params.apply(lambda x: x.get('a')))
    loc_arr = np.array(players.params.apply(lambda x: x.get('loc')))
    scale_arr = np.array(players.params.apply(lambda x: x.get('scale')))
    simulation_arr = skewnorm.rvs(a=a_arr,loc=loc_arr,scale=scale_arr, size=np.array([_SIMULATIONS,a_arr.size]))
    lower_arr, upper_arr = skewnorm.interval(confidence_level_cap, a=a_arr,loc=loc_arr,scale=scale_arr)
    simulation_arr = np.where(simulation_arr < lower_arr, lower_arr, np.where(simulation_arr>upper_arr, upper_arr, simulation_arr))
    return simulation_arr.T


@dataclass
class PlayerData:
    all_players_df: pd.DataFrame
    user_team: UserTeam
    settings: Settings
    player_data: np.array = field(init=False)

    def __post_init__(self):
        self.all_players_df.params = self.all_players_df.params.apply(ast.literal_eval)
        self.player_data = np.array(self.all_players_df)
        draft_pick_arrays = np.split(np.array(self.user_team.draft_picks), len(self.user_team.draft_picks))
        self.available_probability = (1 - norm(np.array(self.all_players_df.loc[:,'adp_sleeper']), np.array(self.all_players_df.loc[:,'sd_adp'])).cdf(draft_pick_arrays)).T
        self.fpts_simulations = player_simulation(self.all_players_df)
        self.index_player_name_map = {player: i for i, player in self.all_players_df.player.to_dict().items()}
        self.column_name_map = {column_name:i for i, column_name in enumerate(self.all_players_df.columns)}
        self.replacement_player_by_position()
        self.adjust_fpts_for_replacement()
        if self.settings.use_injury_risk:
            self.run_games_played_simulation()
            

    def run_games_played_simulation(self):
        n_players = len(self.all_players_df)
        risk_arrays = list(zip(np.split(np.random.uniform(size = [_SIMULATIONS*n_players,17]), n_players), np.split(np.array(self.all_players_df.p_injury_game), n_players)))
        risk_arrays = [player[0] > player[1] for player in risk_arrays]
        self.games_played_simulation = np.vstack([player.sum(axis=1)/17 for player in risk_arrays])
        for idx, position in enumerate(_POSITIONS):
            mask = self.player_data[:,self.column_name_map['position']] == position
            self.fpts_simulations[mask,:] *= self.games_played_simulation[mask,:]

    
    def replacement_player_by_position(self):
        self.replacement_player_fpts = np.zeros(shape=[4,_SIMULATIONS])

        for idx, position in enumerate(_POSITIONS):
            pos_mask = self.player_data[:,self.column_name_map['position']] == position

            if position in ['QB', 'TE']:
                cond1 = self.player_data[:,self.column_name_map['position_rank']] > self.settings.n_starters[position]*self.settings.n_teams
                cond2 = self.player_data[:,self.column_name_map['position_rank']] <= self.settings.n_starters[position]*(self.settings.n_teams+1)
                mask = pos_mask&cond1&cond2
            else:
                flex_starters = (self.settings.n_starters['RB']*self.settings.n_teams 
                                    + self.settings.n_starters['WR']*self.settings.n_teams
                                    + self.settings.n_starters['Flex']*self.settings.n_teams)
                cond3 = self.player_data[:,self.column_name_map['flex_rank']] > flex_starters
                cond4 = self.player_data[:,self.column_name_map['position_rank']] > self.settings.n_starters[position]*self.settings.n_teams
                min_position_rank = self.player_data[:,self.column_name_map['position_rank']][pos_mask&cond3&cond4].min()
                cond5 = self.player_data[:,self.column_name_map['position_rank']] < self.settings.n_teams + min_position_rank
                mask = pos_mask&cond3&cond4&cond5

            fpts_replacements = self.fpts_simulations[mask,:]
            mean = np.mean(fpts_replacements)
            std = np.std(fpts_replacements)
            self.replacement_player_fpts[idx,:] = norm.rvs(loc=mean, scale=std,size=_SIMULATIONS)


    def adjust_fpts_for_replacement(self):
        for idx, position in enumerate(_POSITIONS):
            mask = self.player_data[:,self.column_name_map['position']] == position
            self.fpts_simulations[mask,:] -= self.replacement_player_fpts[idx,:]


    def boolean_mask(self, mask_type:str, players:list = None, columns:list = None, invert_mask:bool = False):
        mask_size = self.player_data.shape[0] if mask_type == 'players' else self.player_data.shape[1]
        mask = np.full(mask_size, False)

        if mask_type == 'players' and players is not None:
            mask = np.full(mask_size, False)
            idx_values = [self.index_player_name_map[player] for player in players]
            mask[np.r_[idx_values]] = True
        elif mask_type == 'columns' and columns is not None:
            mask = np.full(mask_size, False)
            idx_values = [self.column_name_map[column] for column in columns]
            mask[np.r_[idx_values]] = True
        else:
            mask = np.full(mask_size, True)

        if invert_mask:
              mask = np.where(mask==True, False, True)
        return mask


    def get_fpts_simulations(self, players:list = None, invert_mask:bool = False):
        if players is not None:
            return self.fpts_simulations[self.boolean_mask(mask_type = 'players', players=players, invert_mask=invert_mask),:]
        else:
            return self.fpts_simulations
    

    def get_player_data(self, players: list = None, columns:list = None, invert_mask:bool = False):
        if players is not None:
            player_mask = self.boolean_mask(mask_type='players', players=players, invert_mask=invert_mask)
        else:
            player_mask = np.full(self.player_data.shape[0], True)
        if columns is not None:
            column_mask = self.boolean_mask(mask_type='columns', columns =columns)
        else:
            column_mask = np.full(self.player_data.shape[1], True)

        temp_arr = self.player_data[player_mask,:]
        return temp_arr[:,column_mask]


    def get_player_availability(self, players:list = None, column_nums:list = None, invert_mask: bool = False):
        if column_nums == None:
            column_nums = range(0,self.available_probability.shape[1])
        if players == None:
            return self.available_probability[:,np.r_[column_nums]]
        else:
            player_mask = self.boolean_mask('players',players,invert_mask=invert_mask)
            temp_arr = self.available_probability[player_mask,:]
            return temp_arr[:,np.r_[column_nums]]


@dataclass
class PositionalDistributions:
    player_data: PlayerData
    positional_distributions: dict = field(init=False)

    def __post_init__(self):
        self.reset_positional_distributions()

    
    def reset_positional_distributions(self):
        self.positional_distributions = {}
        for i in range(1,17):
            self.positional_distributions[i] = {}


    def players_available_in_round(self, drafted_players: list, draft_round: int):
        round_availability = self.player_data.get_player_availability(drafted_players, column_nums=[draft_round-1], invert_mask=True)
        players = self.player_data.get_player_data(drafted_players, columns=['player'], invert_mask=True)
        return list(players[round_availability< 0.9999])


    def get_positional_distributions(self, draft_pick:int, players_available:list):
        simulations = _SIMULATIONS
        round_distributions = {}
        pos_arr = self.player_data.get_player_data(players=players_available,columns=['position'])
        adp_arr = self.player_data.get_player_data(players=players_available,columns=['adp_sleeper']) - draft_pick
        sd_adp_arr = self.player_data.get_player_data(players=players_available,columns=['sd_adp'])
        points = self.player_data.get_fpts_simulations(players=players_available)
        drafted_mask = np.where(norm.rvs(adp_arr, sd_adp_arr, size=np.array([adp_arr.size, simulations]))>0,True,False)
        for position in _POSITIONS:
            position_mask = (pos_arr == position)
            masked_points = np.where(np.array(drafted_mask[position_mask.T[0],:]), np.array(points[position_mask.T[0],:]), 0).max(axis=0)
            round_distributions[position] = {'mean': round(masked_points.mean(),4), 'sd': round(masked_points.std(),4)}

        return round_distributions


    def run(self, drafted_players: list, current_round:int, draft_picks: list):
        self.reset_positional_distributions()
        self.available_players_by_round = []
        for draft_round, draft_pick in enumerate(draft_picks):
            if current_round < draft_round and draft_round < 9:
                players_available = self.players_available_in_round(drafted_players, draft_round)
                self.available_players_by_round.append({draft_round: players_available})
                self.positional_distributions[draft_round] = self.get_positional_distributions(draft_pick, players_available)
            else:
                self.positional_distributions[draft_round] = {'QB': {'mean': 0, 'sd': 0},
                                                                'RB': {'mean': 0, 'sd': 0},
                                                                'WR': {'mean': 0, 'sd': 0},
                                                                'TE': {'mean': 0, 'sd': 0}}

        

@dataclass
class DraftSimulation:
    settings: Settings 
    draft_state: DraftState
    user_team: UserTeam
    distributions: PositionalDistributions
    player_data: PlayerData
    draft_combinations: np.array = field(init=False)

    def __post_init__(self):
        self.generate_draft_combos()


    def generate_draft_combos(self):
        starting_positions = [position for position, n_start in self.settings.n_starters.items() for n_start in range(n_start)]
        initial_combos = np.array(list(set(itertools.permutations(starting_positions))))
        flex_idxs = np.where(initial_combos=='Flex')
        arrs = []
        for product in list(itertools.product(['RB', 'WR'], repeat=self.settings.n_starters['Flex'])):
            arr_combo = np.array(initial_combos.copy())
            for idx in range(0, self.settings.n_starters['Flex']):
                arr_combo[flex_idxs[0][idx::self.settings.n_starters['Flex']], flex_idxs[1][idx::self.settings.n_starters['Flex']]] = product[idx]
            arrs.append(arr_combo)
        self.draft_combinations = np.vstack(arrs)


    def update_distributions(self):
        self.distributions.run(self.draft_state.drafted_players, 
                               self.draft_state.current_round,
                               self.user_team.draft_picks
                               )


    def user_drafted_positions_order(self):
        return [player['position'] for player in self.user_team.players_drafted]
    

    def user_drafted_players_simulation(self):
        if bool(self.user_team.players_drafted):
            players = [player['player'] for player in self.user_team.players_drafted]
            return self.player_data.get_fpts_simulations(players).sum(axis=0)
        else:
            return 0
        

    def starting_positions_needed(self):
        positional_needs = []
        for position in _POSITIONS:
            if self.user_team.starters_needed[position]>0: 
                positional_needs.append(position)
            elif (self.user_team.starters_needed['Flex']>0) & (position in ['RB', 'WR']):
                positional_needs.append(position)
        self.positional_needs = positional_needs
    

    def positional_params(self, position: str):
        size = np.shape(self.draft_combinations)[1]
        means = np.zeros([size])
        sds = np.zeros([size])

        for draft_round in range(size):
            means[draft_round] = 0.0 if self.draft_state.current_round > draft_round else self.distributions.positional_distributions[draft_round][position]['mean']
            sds[draft_round] = 0.0 if self.draft_state.current_round > draft_round else self.distributions.positional_distributions[draft_round][position]['sd']

        return means, sds
    

    def get_remaining_combinations(self):
        if bool(self.user_team.players_drafted):
            positions_drafted_order = [player['position'] for player in self.user_team.players_drafted]
            combination_valid = np.all(self.draft_combinations[:,0:self.draft_state.current_round-1] == np.array(positions_drafted_order), axis=1)
            combinations_remaining = self.draft_combinations[combination_valid]
        else:
            combinations_remaining = self.draft_combinations
        return combinations_remaining
    

    def weighted_average_remaining_combinations(self, means: np.array, sds: np.array, draft_combinations: np.array):
        weighted_avgs = {}
        
        for position in self.positional_needs:
            rows_in_combinations = draft_combinations[:,self.draft_state.current_round-1] == position
            position_means = means[rows_in_combinations,:].sum(axis=1)
            position_variance = np.sum(sds[rows_in_combinations]**2, axis=1)

            order_of_best_means = position_means.argsort()[::-1]
            weights = 1/np.exp(np.arange(0.1,5.0,(5.0-0.1)/(position_means.size)))
            diff = weights.size - position_means.size
            if diff!=0:
                weights = weights[:-diff]
            weighted_avg_means = np.sum(position_means[order_of_best_means]*weights)/np.sum(weights)
            weighted_avg_sds = np.sum(position_variance[order_of_best_means]*weights)/np.sum(weights)
            weighted_avg_sds = np.sqrt(weighted_avg_sds)

            weighted_avgs[position] = {'mean': weighted_avg_means, 'sd': weighted_avg_sds}
        return weighted_avgs
    

    def remaining_combination_params(self):
        combinations_remaining = self.get_remaining_combinations()

        # Initialize arrays
        combination_means = np.zeros(combinations_remaining.shape)
        combination_sds = np.zeros(combinations_remaining.shape)

        # add positional paramaters to the remaining draft combinations array
        for position in _POSITIONS:
            means, sds = self.positional_params(position)
            combination_means = np.where(combinations_remaining==position, means, combination_means)
            combination_sds = np.where(combinations_remaining==position, sds, combination_sds)

        self.weighted_avgs_combinations = self.weighted_average_remaining_combinations(combination_means, combination_sds, combinations_remaining)


    def get_players_for_simulation(self, position: str):
        cond1 = self.player_data.all_players_df.position == position
        cond2 = self.player_data.all_players_df.player.isin(self.draft_state.drafted_players)
        cond3 = self.player_data.all_players_df.adp_sleeper < self.draft_state.current_pick + self.settings.n_teams*6
        return list(self.player_data.all_players_df[(cond1)&~(cond2)&(cond3)].player)
    
    
    def compute_best_starter(self):
        simulations = _SIMULATIONS

        cols = ['player', 'position', 'team', 'bye', 'points', 'floor', 'ceiling']
        dfs = []
        player_simulated_points = []

        if len(self.positional_needs)>1:
            for position in self.positional_needs:
                players = self.get_players_for_simulation(position)
                points = self.player_data.get_fpts_simulations(players=players)
                drafted_players_points = self.user_drafted_players_simulation()
                position_combinations_simulation = norm.rvs(self.weighted_avgs_combinations[position]['mean'], self.weighted_avgs_combinations[position]['sd'], simulations)
                p_not_available_2rds = 1 - self.player_data.get_player_availability(players=players,column_nums=[self.draft_state.current_round+1])
                player_combined = (points*p_not_available_2rds + position_combinations_simulation + drafted_players_points)
                player_simulated_points.append(player_combined)
                dfs.append(pd.DataFrame(self.player_data.get_player_data(players=players, columns=cols),columns=cols))
        else:
            for position in _POSITIONS:
                players = self.get_players_for_simulation(position)
                points = self.player_data.get_fpts_simulations(players=players)
                p_not_available_2rds = 1 - self.player_data.get_player_availability(players=players,column_nums=[self.draft_state.current_round+1])
                player_simulated_points.append(points*p_not_available_2rds)
                dfs.append(pd.DataFrame(self.player_data.get_player_data(players=players, columns=cols),columns=cols))
            
        self.player_lift_df = pd.concat(dfs)
        stacked_player_simulations = np.vstack(player_simulated_points)
        best_player_simulation = stacked_player_simulations.max(axis=0)
        second_best_player_simulation = np.where(stacked_player_simulations == best_player_simulation, 0, stacked_player_simulations).max(axis=0)
        self.player_lift_df['pBestValue'] = (stacked_player_simulations == stacked_player_simulations.max(axis=0)).sum(axis=1)/simulations
        self.player_lift_df['eValueAdded'] = np.nanmean(np.where(stacked_player_simulations == best_player_simulation, stacked_player_simulations - second_best_player_simulation, np.nan), axis=1)
        self.player_lift_df['eValueLost'] = np.nanmean(np.where(stacked_player_simulations == best_player_simulation, np.nan, stacked_player_simulations - best_player_simulation), axis=1)
        self.player_lift_df.sort_values(by='pBestValue', ascending=False, inplace=True)
        self.player_lift_df['eValue'] = round(self.player_lift_df['eValueAdded'] + self.player_lift_df['eValueLost'],1)
        self.player_lift_df = self.player_lift_df[self.player_lift_df.pBestValue>=0.0001]
        self.player_lift_df = self.player_lift_df[['player', 'position', 'pBestValue', 'eValue']]
        self.player_lift_df.reset_index(inplace=True, drop=True)
        


    def run(self):
        self.starting_positions_needed()
        if len(self.positional_needs)>1:
            self.update_distributions()
            self.remaining_combination_params()
        self.compute_best_starter()



###### TO DO ######
# 1. Remove hard coding of adp_sleepr