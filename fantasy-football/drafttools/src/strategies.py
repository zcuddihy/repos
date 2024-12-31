import numpy as np

from src.constants import POSITION_IDX_MAP, POSITIONS

# import pandas as pd

STRATEGY_KWARGS = {
    "best_lineup": {},
    "best_bench": {},
    "value_over_last_starter": {},
    "value_over_replacement": {},
    "value_over_next_round": {},
}


def add_simulations_to_combinations(
    combinations: np.ndarray,
    position_expected_value: dict[str, np.ndarray],
    replace_starter_positions_with_nan: bool,
    replace_bench_with_nan: bool,
):
    position_mapping_for_combinations = np.vectorize(POSITION_IDX_MAP.get)(combinations)
    n_positions_in_combinations = position_mapping_for_combinations.shape[1]
    simulated_points_array = flatten_postion_expected_value_to_array(
        position_expected_value
    )
    combinations_with_simulations = simulated_points_array[
        position_mapping_for_combinations, np.arange(n_positions_in_combinations)
    ]

    if replace_bench_with_nan:
        combinations_with_simulations = np.where(
            (combinations == "BENCH")[:, :, np.newaxis],
            np.full(combinations_with_simulations.shape, np.nan),
            combinations_with_simulations,
        )
    elif replace_starter_positions_with_nan:
        combinations_with_simulations = np.where(
            (combinations != "BENCH")[:, :, np.newaxis],
            np.full(combinations_with_simulations.shape, np.nan),
            combinations_with_simulations,
        )

    # [:, 0] is for the current round of all possible combinations
    # [:, 1:] removes the current round from the combinations
    combinations_with_simulations = np.nansum(
        combinations_with_simulations[:, 1:], axis=1
    )
    position_for_simulated_combination = combinations[:, 0]
    return position_for_simulated_combination, combinations_with_simulations


def position_best_combination_with_simulations(
    position_for_simulated_combination: np.ndarray,
    combinations_with_simulations: np.ndarray,
):
    best_combination_by_position = {}
    for position in POSITIONS:
        if position in position_for_simulated_combination:
            combinations_for_position = combinations_with_simulations[
                position_for_simulated_combination == position
            ]
            highest_simulations = combinations_with_simulations[
                position_for_simulated_combination == position
            ].max(axis=0)

            idx_of_best_combination = np.argmax(
                (combinations_for_position == highest_simulations).sum(axis=1)
            )
            best_combination_by_position[position] = combinations_for_position[
                idx_of_best_combination
            ]
    return best_combination_by_position


def best_combination_with_available_players():
    pass


def flatten_postion_expected_value_to_array(
    position_expected_value: dict[str, np.ndarray],
) -> np.ndarray:
    position_expected_value_array = np.array(
        [
            position_expected_value[pos]
            for pos in POSITION_IDX_MAP
            if pos in position_expected_value.keys()
        ]
    )
    return position_expected_value_array


# @dataclass
# class Strategies:
#     n_simulations: int
#     player_z_scores: np.ndarray = field(init=False)

#     def run(
#         self,
#         strategies_for_pick: list[str],
#         player_data: pd.DataFrame,
#         user_draft_picks: list[str],
#         roster_settings: dict[str, int],
#         position_team_needs: dict[str, int],
#         flex_positions: list[str],
#         num_teams: int,
#         positions_allowed_on_bench: list[str],
#     ):
#         simulator = Simulations(self.n_simulations)
#         simulator.run_simulations(
#             player_data,
#             user_draft_picks,
#             roster_settings,
#             flex_positions,
#             num_teams,
#             positions_allowed_on_bench,
#         )
#         draft_combinations = simulate_all_remaining_position_combinations(
#             position_team_needs, flex_positions, positions_allowed_on_bench
#         )

#     def _add_simulations_to_combinations(
#         self, use_baselines_fpts: bool, replace_bench_with_nan: bool
#     ):
#         position_mapping_for_combinations = np.vectorize(POSITION_IDX_MAP.get)(
#             self._postion_draft_combinations
#         )
#         n_positions_in_combinations = position_mapping_for_combinations.shape[1]
#         simulated_points_array = self._get_array_position_fpts_all_rounds(
#             use_baselines_fpts
#         )
#         combinations_with_simulations = simulated_points_array[
#             position_mapping_for_combinations, np.arange(n_positions_in_combinations)
#         ]

#         if (replace_bench_with_nan) and (len(self._bench_positions) > 0):
#             combinations_with_simulations = np.where(
#                 np.isin(self._postion_draft_combinations, self._bench_positions)[
#                     :, :, np.newaxis
#                 ],
#                 np.full(combinations_with_simulations.shape, np.nan),
#                 combinations_with_simulations,
#             )

#         # [:, 0] is for the current round of all possible combinations
#         # [:, 1:] removes the current round from the combinations
#         combinations_with_simulations = np.nansum(
#             combinations_with_simulations[:, 1:], axis=1
#         )
#         position_for_simulated_combination = self._postion_draft_combinations[:, 0]
#         return position_for_simulated_combination, combinations_with_simulations
