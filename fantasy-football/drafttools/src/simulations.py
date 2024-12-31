import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.constants import POSITION_IDX_MAP, POSITIONS

Z_SCORE = 1.96


def simulate_all_remaining_position_combinations(
    position_team_needs: dict[str, int],
    flex_positions: list[str],
    positions_allowed_on_bench: list[str],
):
    positions_for_combinations = []
    for position, num_starters_needed in position_team_needs.items():
        if position not in ["FLEX", "K", "DST", "BENCH"] and num_starters_needed > 0:
            for _ in range(num_starters_needed):
                positions_for_combinations.append(position)

    positions_for_combinations = add_bench_to_position_combinations(
        positions_for_combinations, position_team_needs
    )

    if position_team_needs["FLEX"] > 0:
        flex_combinations = list(
            itertools.product(flex_positions, repeat=position_team_needs["FLEX"])
        )
        positions_for_combinations = [
            positions_for_combinations + list(flex) for flex in flex_combinations
        ]
        flat_list = list(
            itertools.chain(
                *[itertools.permutations(combo) for combo in positions_for_combinations]
            )
        )
        simulated_combinations = np.unique(list(set(flat_list)), axis=0)
    else:
        simulated_combinations = np.unique(
            list(set(itertools.permutations(positions_for_combinations))), axis=0
        )
    simulated_combinations = check_validity_of_combinations(
        simulated_combinations, positions_allowed_on_bench
    )
    return simulated_combinations


def add_bench_to_position_combinations(
    positions_for_combinations: list,
    position_team_needs: dict[str, int],
    max_length_combinations: int = 8,
):
    """_summary_

    Args:
        positions_for_combinations (list): _description_
        position_team_needs (dict[str, int]): _description_
        max_length_combinations (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    num_bench_needed = position_team_needs["BENCH"]
    num_flex_needed = position_team_needs["FLEX"]
    num_starters_in_combinations = len(positions_for_combinations)

    if (num_starters_in_combinations + num_flex_needed < max_length_combinations) and (
        num_bench_needed > 0
    ):
        num_bench_to_add = np.min(
            (
                max_length_combinations
                - (num_starters_in_combinations + num_flex_needed),
                num_bench_needed,
            )
        )
        for _ in range(num_bench_to_add):
            positions_for_combinations.append("BENCH")
    return positions_for_combinations


def check_validity_of_combinations(
    positions_for_combinations, positions_allowed_on_bench: list
):
    """Check if each possible draft combination is a valid draft sequence.

    When the combinations are generated with BENCH positions added, it's possible
    some combinations generate a BENCH position before it's eligibile to be filled.

    Args:
        positions_for_combinations (np.ndarray): _description_
        positions_allowed_on_bench (list): _description_

    Returns:
        positions_for_combinations (np.ndarray): _description_
    """
    bench_locs = positions_for_combinations == "BENCH"
    valid_locations = []
    for position in positions_allowed_on_bench:
        bench_eligible_loc = (positions_for_combinations == position).cumsum(
            axis=1
        ) == (positions_for_combinations == position).sum(axis=1).reshape(-1, 1)
        is_eligible_combination = np.logical_and(bench_locs, bench_eligible_loc).sum(
            axis=1
        ) == bench_locs.sum(axis=1)
        valid_locations.append(is_eligible_combination)

    all_valid_locations = np.all(np.vstack(valid_locations), axis=0)
    return positions_for_combinations[all_valid_locations]


def cap_simulations_at_z_score(
    array: np.ndarray,
    means: np.ndarray,
    std_devs: np.ndarray,
    z_score_cap: float = Z_SCORE,
):
    max_vals = means + z_score_cap * std_devs
    min_vals = means - z_score_cap * std_devs
    array = np.where(
        array > max_vals,
        max_vals,
        np.where(array < min_vals, min_vals, array),
    )
    return array


@dataclass
class Simulations:
    n_simulations: int
    player_position_labels: np.ndarray = field(init=False)
    player_fpts: np.ndarray = field(init=False)
    player_fpts_with_baseline: np.ndarray = field(init=False)
    player_availability_by_round: np.ndarray = field(init=False)
    player_availability_next_round_pct: np.ndarray = field(init=False)
    player_adp: np.ndarray = field(init=False)
    position_baseline_starters: dict[str, np.ndarray] = field(init=False)
    position_baseline_replacements: dict[str, np.ndarray] = field(init=False)
    position_expected_value_per_round: dict[str, np.ndarray] = field(init=False)
    position_baseline_expected_value_per_round: dict[str, np.ndarray] = field(
        init=False
    )

    def run_simulations(
        self,
        player_data: pd.DataFrame,
        user_draft_picks: list,
        roster_settings: dict[str, int],
        flex_positions: list[str],
        num_teams: int,
        positions_allowed_on_bench: list[str],
    ):
        num_picks_user_made = player_data.is_user_draft_pick.sum()
        drafted_player_mask = np.array(player_data.drafted)
        self.player_position_labels = np.array(player_data.position)

        self.player_fpts = self.__simulate_player_fpts(player_data)
        self.player_adp = self.__simulate_adp(player_data)
        self.player_availability_by_round = self.__simulate_player_availability(
            user_draft_picks, num_picks_user_made, drafted_player_mask
        )
        self.player_availability_next_round_pct = (
            self.player_availability_by_round[:, 1].sum(axis=1) / self.n_simulations
        )
        self.position_baseline_starters = self.__get_baseline_starters(
            num_teams, flex_positions, roster_settings
        )
        self.player_fpts_with_baseline = self.__adjust_player_fpts_with_baseline()
        self.position_expected_value_per_round = (
            self.__get_position_expected_value_per_round(
                False, positions_allowed_on_bench
            )
        )
        self.position_baseline_expected_value_per_round = (
            self.__get_position_expected_value_per_round(
                True, positions_allowed_on_bench
            )
        )

    def calculate_value_over_next_round(self, use_position_baseline: bool):
        if use_position_baseline:
            tmp_player_fpts = self.player_fpts_with_baseline
            tmp_position_fpts_baseline = self.position_baseline_expected_value_per_round
        else:
            tmp_player_fpts = self.player_fpts
            tmp_position_fpts_baseline = self.position_expected_value_per_round

        player_position_mapping = np.vectorize(POSITION_IDX_MAP.get)(
            self.player_position_labels
        )
        position_value_next_round = np.vstack(
            [tmp_position_fpts_baseline[position][1] for position in POSITIONS]
        )
        value_over_next_round = (
            tmp_player_fpts - position_value_next_round[player_position_mapping]
        )
        return value_over_next_round

    def __simulate_player_fpts(self, player_data: pd.DataFrame):
        # simulation of fantasy points
        means = np.array(player_data.fpts_mean).reshape(-1, 1)
        stds = np.array(player_data.fpts_stddev).reshape(-1, 1)
        sim_fpts = np.random.normal(
            loc=means, scale=stds, size=[len(means), self.n_simulations]
        )
        # Winzsore values to be within the 95th percentile
        sim_fpts = cap_simulations_at_z_score(sim_fpts, means, stds)
        return sim_fpts

    def __simulate_adp(self, player_data: pd.DataFrame):
        means = np.array(player_data.adp).reshape(-1, 1)
        stds = np.array(player_data.adp_stddev).reshape(-1, 1)
        sim_adp = np.round(
            np.random.normal(
                loc=means, scale=stds, size=[len(means), self.n_simulations]
            ),
            0,
        )
        sim_adp = np.where(sim_adp < 1, 1, sim_adp)
        # Winzsore values to be within the 95th percentile
        sim_adp = cap_simulations_at_z_score(sim_adp, means, stds)
        return sim_adp

    def __simulate_player_availability(
        self,
        user_draft_picks: list,
        num_picks_user_made: int,
        drafted_player_mask: np.ndarray,
    ):
        # simulation of if the player is available in a given round
        # [num_picks_user_made:] removes the previous draft picks that have been made
        sim_available_by_round = self.player_adp[:, :, np.newaxis] > np.array(
            user_draft_picks[num_picks_user_made:]
        )
        # Transpose the second axis so each row within
        # a player corresponds to a draft round
        sim_available_by_round = sim_available_by_round.transpose(0, 2, 1)
        # Adjust players for current round to all be true if available
        # and false if they have been drafted
        sim_available_by_round[:, 0] = True
        sim_available_by_round[drafted_player_mask, :, :] = False
        return sim_available_by_round

    def __baseline_starter_at_position(
        self,
        position: str,
        num_teams: int,
        roster_settings: dict,
    ):
        n_starters = num_teams * roster_settings[position]
        position_arr = self.player_fpts[self.player_position_labels == position]
        return np.sort(position_arr, axis=0)[-n_starters, :]

    def __possible_flex_starters_at_position(
        self, position: str, num_teams: int, roster_settings: dict
    ):
        n_starters_position = num_teams * roster_settings[position]
        n_starters_flex = num_teams * roster_settings["FLEX"]
        position_arr = self.player_fpts[self.player_position_labels == position]
        flex_starters = np.sort(position_arr, axis=0)[
            -(n_starters_flex + n_starters_position) : -n_starters_position, :
        ]
        position_labels = (
            np.array([position]).repeat(flex_starters.shape[0]).reshape(-1, 1)
        )
        return position_labels, flex_starters

    def __adjust_baselines_for_flex(
        self,
        baseline_starter: dict,
        num_teams: int,
        flex_positions: list,
        roster_settings: dict,
    ):
        baseline_starter = baseline_starter
        n_starters_flex = num_teams * roster_settings["FLEX"]
        possible_flex_starters = {
            position: self.__possible_flex_starters_at_position(
                position, num_teams, roster_settings
            )
            for position in flex_positions
        }
        possible_flex_starters_fpts = np.vstack(
            [v[1] for _, v in possible_flex_starters.items()]
        )
        possible_flex_starters_pos_labels = np.vstack(
            [v[0] for _, v in possible_flex_starters.items()]
        ).flatten()
        baseline_flex = np.sort(possible_flex_starters_fpts, axis=0)
        baseline_flex = baseline_flex[-n_starters_flex]
        is_a_flex_starter = possible_flex_starters_fpts >= baseline_flex
        for position in flex_positions:
            adjusted_baseline = np.min(
                np.where(
                    is_a_flex_starter[possible_flex_starters_pos_labels == position, :],
                    possible_flex_starters_fpts[
                        possible_flex_starters_pos_labels == position, :
                    ],
                    1e6,
                ),
                axis=0,
            )
            baseline_starter[position] = np.min(
                (baseline_starter[position], adjusted_baseline), axis=0
            )
        return baseline_starter

    def __get_baseline_starters(
        self,
        num_teams: int,
        flex_positions: list,
        roster_settings: dict,
    ):
        baseline_starter = {
            position: self.__baseline_starter_at_position(
                position, num_teams, roster_settings
            )
            for position in POSITIONS
        }
        if roster_settings["FLEX"] > 0:
            baseline_starter = self.__adjust_baselines_for_flex(
                baseline_starter,
                num_teams,
                flex_positions,
                roster_settings,
            )
        return baseline_starter

    def __adjust_player_fpts_with_baseline(self):
        position_mapping = np.vectorize(POSITION_IDX_MAP.get)(
            self.player_position_labels
        )
        baseline_array = np.vstack(
            [self.position_baseline_starters[position] for position in POSITIONS]
        )[position_mapping]
        return self.player_fpts - baseline_array

    def __get_position_expected_value_per_round(
        self, use_baselines_fpts: bool, positions_allowed_on_bench: list[str]
    ):
        if use_baselines_fpts:
            player_fpts_for_expected_val = self.player_fpts_with_baseline
        else:
            player_fpts_for_expected_val = self.player_fpts
        # mask for simulated points if player is available
        # in the future round(s)
        sim_fpts_masked = np.where(
            self.player_availability_by_round,
            player_fpts_for_expected_val[:, np.newaxis, :],
            np.nan,
        )

        position_expected_value_per_round = {}
        for position in POSITIONS:
            positions_idxs = self.player_position_labels == position
            position_expected_value_per_round[position] = np.nanmax(
                sim_fpts_masked[positions_idxs], axis=0
            )

        position_expected_value_per_round["BENCH"] = np.nanmax(
            sim_fpts_masked[
                np.isin(self.player_position_labels, positions_allowed_on_bench)
            ],
            axis=0,
        )

        return position_expected_value_per_round
