from collections import Counter

import pandas as pd
import requests


class Settings:
    def __init__(
        self,
        user_draft_slot: int,
        sleeper_draft_id: str,
        roster_settings: dict[str, int],
        flex_positions: list[str],
        num_teams: int,
        use_third_round_reversal: bool,
        positions_allowed_on_bench: list[str],
    ):
        self._sleeper_draft_id = sleeper_draft_id
        self._user_draft_slot = user_draft_slot
        self._roster_settings = roster_settings
        self._flex_positions = flex_positions
        self._num_teams = num_teams
        self._use_third_round_reversal = use_third_round_reversal
        self._positions_allowed_on_bench = positions_allowed_on_bench

    def _get_sleeper_draft_picks(self):
        url = f"https://api.sleeper.app/v1/draft/{self._sleeper_draft_id}/picks"
        r = requests.get(url)
        return r.text


class UserTeam(Settings):
    def __init__(
        self,
        user_draft_slot: int,
        sleeper_draft_id: str,
        roster_settings: dict[str, int],
        flex_positions: list[str],
        num_teams: int,
        use_third_round_reversal: bool,
        positions_allowed_on_bench: list[str],
    ):
        super().__init__(
            user_draft_slot,
            sleeper_draft_id,
            roster_settings,
            flex_positions,
            num_teams,
            use_third_round_reversal,
            positions_allowed_on_bench,
        )
        self._user_draft_picks = [
            self._team_draft_picks(round)
            for round in range(1, sum(roster_settings.values()) + 1, 1)
        ]
        self._user_positions_drafted: dict[str, int] = {}
        self._position_team_needs = self._roster_settings

    def update_team_state(self, player_data):
        self._user_positions_drafted = self._get_user_drafted_positions(player_data)
        self._position_team_needs = self._get_position_team_needs()

    def _team_draft_picks(self, draft_round: int):
        if (not self._use_third_round_reversal) or (draft_round < 3):
            return self._num_teams * (draft_round - 1) + (
                self._user_draft_slot
                if draft_round % 2 == 1
                else self._num_teams - self._user_draft_slot + 1
            )
        else:
            return self._num_teams * (draft_round - 1) + (
                self._user_draft_slot
                if draft_round % 2 == 0
                else self._num_teams - self._user_draft_slot + 1
            )

    def _get_user_drafted_positions(self, player_data: pd.DataFrame):
        _user_positions_drafted = (
            player_data.copy()
            .sort_values(by="picked_at")[player_data.is_user_draft_pick]
            .position.to_list()
        )
        _user_positions_drafted = dict(Counter(_user_positions_drafted))
        return _user_positions_drafted

    def _get_position_team_needs(self):
        position_team_needs = {}
        for pos, n_starters in self._roster_settings.items():
            if pos not in ["FLEX", "K", "DST", "BENCH"]:
                if pos in self._user_positions_drafted.keys():
                    position_team_needs[pos] = (
                        n_starters - self._user_positions_drafted[pos]
                    )
                else:
                    position_team_needs[pos] = n_starters
        if any(position_team_needs[pos] < 0 for pos in self._flex_positions):
            total_flex_drafted = sum(
                [
                    -1 * position_team_needs[position]
                    for position in self._flex_positions
                    if position_team_needs[position] < 0
                ]
            )
            total_flex_allowed = self._roster_settings["FLEX"]
            position_team_needs["FLEX"] = total_flex_allowed - total_flex_drafted
        else:
            position_team_needs["FLEX"] = self._roster_settings["FLEX"]
        return position_team_needs


class Draft:
    pass
