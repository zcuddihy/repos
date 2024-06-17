import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from draft import Settings, UserTeam, DraftState, PositionalDistributions, DraftSimulation, PlayerData

st.set_page_config(layout="wide")
col1, col2 = st.columns([.25,.75], gap='medium')
_POSITION_COLORS = {'QB': 'rgba(239, 116, 161, 0.8)', #'#d9534f',
                    'RB': 'rgba(143, 242, 202, 0.8)', #'#5cb85c',
                    'WR': 'rgba(86, 201, 248, 0.8)', #'#ffc107',
                    'TE': 'rgba(254, 174, 88, 0.8)' #'#4c9be8'
                    }

_BYE_OVERLAP_COLOR = '#d9534f'

def initialization():
    if 'draft_started' not in st.session_state:
        st.session_state.draft_started = True
        st.session_state.df = pd.read_csv('merged_data.csv')
        st.session_state.df['Drafted?'] = False
        st.session_state.drafted_players = []
        st.session_state.is_user_pick = False
        
        kwargs = {
            'n_teams':12,
            'n_starters': {'QB':1, 'RB':2, 'WR':2, 'TE':1, 'Flex': 2},
            'third_round_reversal': False,
            'roster_size': 14,
            'adp_source': 'sleeper',
            'use_injury_risk': True
        }
        st.session_state.settings = Settings(**kwargs)
        st.session_state.user_team =UserTeam(st.session_state.settings,7)
        st.session_state.player_data = PlayerData(pd.read_csv('merged_data.csv'), st.session_state.user_team, st.session_state.settings)
        st.session_state.distributions = PositionalDistributions(st.session_state.player_data)
        st.session_state.draft_state = DraftState(pd.read_csv('merged_data.csv'), st.session_state.settings, st.session_state.user_team)
        st.session_state.simulation = DraftSimulation(
            st.session_state.settings,
            st.session_state.draft_state,
            st.session_state.user_team,
            st.session_state.distributions,
            st.session_state.player_data
        )

def log_draft_pick():
    player_index=list(st.session_state["data_editor"]['edited_rows'].keys())[0]
    player_name = df.player.loc[player_index]
    st.session_state.drafted_players.append(player_name)
    st.session_state.draft_state.update_draft_state(st.session_state.drafted_players)


def undo_pick():
    if st.session_state.draft_state.current_pick != 1:
        del st.session_state.drafted_players[-1]
        st.session_state.draft_state.undo_last_pick()


def is_user_pick():
    if st.session_state.draft_state.current_pick in st.session_state.user_team.draft_picks:
        st.session_state.simulation.run()


def dataframe_for_display():
    col_order = ['Drafted?', 'player', 'position', 'team', 'bye', 'adp_sleeper', 'pBestValue', 'eValue', 'points', 'injury_risk']
    df = st.session_state.df[~st.session_state.df.player.isin(st.session_state.drafted_players)].reset_index(drop=True)[['Drafted?', 'player', 'position', 'team', 'points', 'adp_sleeper', 'bye', 'injury_risk']]
    df.sort_values(by='adp_sleeper', ascending=True)
    try:
        simulation_df = st.session_state.simulation.player_lift_df.copy()
        df = df.merge(simulation_df, on = ['player', 'position'], how='left')
        df.pBestValue.fillna(0,inplace=True)
        df.eValue.fillna(df.eValue.min(), inplace=True)
    except:
        df[['pBestValue', 'eValue']] = '--'

    if st.session_state.draft_state.current_pick in st.session_state.user_team.draft_picks:
        df.sort_values(by='pBestValue', ascending=False, inplace=True)
        return df[col_order].reset_index(drop=True)
    else:
        df.sort_values(by='adp_sleeper', ascending=True, inplace=True)
        return df[col_order].reset_index(drop=True)
    

def ChangeButtonColour(widget_label, font_color, background_color='transparent'):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}'
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)

initialization()
is_user_pick()

col2a, col2b, col2c= col2.columns(3)

col2a.metric(label="Current Round", value=st.session_state.draft_state.current_round)
col2b.metric(label="Current Pick", value=st.session_state.draft_state.current_pick)

if col2c.button('Undo Last Pick'):
    undo_pick()

df = dataframe_for_display()
df_style = df.style.apply(lambda r: [f"background-color:{_POSITION_COLORS.get(r['position'],'')}; color: black"]*len(r), axis=1)

col2.data_editor(
    df_style,
    column_config={
        "player_drafted": st.column_config.CheckboxColumn(
            "Drafted?",
            default=False,
            width="medium"
        )
    },
    hide_index=True,
    key="data_editor",
    disabled = ['player', 'position'],
    on_change=log_draft_pick,
    use_container_width=True,
)

settings = col1.expander("Settings")
with settings:
    sleeper_league_id = st.text_input('Sleeper Leauge ID', disabled = True if st.session_state.draft_started else False)
    col1s, col2s = settings.columns(2)
    n_teams = col1s.number_input('Number of Teams', min_value=6, max_value=16, value=12, step=1, key='n_teams', disabled = True if st.session_state.draft_started else False)
    user_team  = col1s.selectbox(
                'Your Team',
                (f'Team {n_team}' for n_team in range(1,n_teams+1)),
                key='user_team_id',
                disabled = True if st.session_state.draft_started else False)
    ppr = col2s.selectbox(
                'PPR',
                (0,0.5,1.0),
                key='ppr_type',
                disabled = True if st.session_state.draft_started else False)
    
    third_round_reversal = col2s.selectbox(
                'Third Round Reversal',
                (True, False),
                disabled = True if st.session_state.draft_started else False
                )

    n_qb = col1s.number_input('QB', min_value=1, max_value=2, value=1, step=1, key='n_QB',disabled = True if st.session_state.draft_started else False)
    n_rb = col1s.number_input('RB', min_value=0, max_value=4, value=2, step=1, key='n_RB',disabled = True if st.session_state.draft_started else False)
    n_wr = col1s.number_input('WR', min_value=0, max_value=4, value=2, step=1, key='n_WR',disabled = True if st.session_state.draft_started else False)
    n_te = col2s.number_input('TE', min_value=0, max_value=2, value=1, step=1, key='n_TE',disabled = True if st.session_state.draft_started else False)
    n_flex = col2s.number_input('Flex', min_value=0, max_value=4, value=1, step=1, key='n_Flex',disabled = True if st.session_state.draft_started else False)
    n_bench = col2s.number_input('Bench', min_value=0, max_value=8, value=5, step=1, key='n_Bench',disabled = True if st.session_state.draft_started else False)

    start_draft = st.button("Start Draft", key='start-draft', disabled = True if st.session_state.draft_started else False, use_container_width=True)
    reset_draft = st.button("Reset Draft", key='reset-draft', disabled = False if st.session_state.draft_started else True, use_container_width=True)
    if st.button('Sync League', disabled = True if st.session_state.draft_started else False, use_container_width=True):
        pass
    #ChangeButtonColour("Start Draft", 'white', '#5cb85c')
    #ChangeButtonColour("Reset Draft", 'white', '#d9534f')
    
col1.write(st.session_state.user_team.players_drafted)

