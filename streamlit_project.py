import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# --- 0. グローバル設定 ---
st.set_page_config(layout="wide", page_title="J.League Physical Dashboard 2025")
st.subheader('All data by SkillCorner')

# --- 1. 共通関数 ---

def to_excel(df: pd.DataFrame):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Ranking Data')
    processed_data = output.getvalue()
    return processed_data

@st.cache_data(ttl=900)
def get_data(league_key):
    file_name = LEAGUE_FILE_MAP.get(league_key)
    file_path = f"data/{file_name}"
    try:
        with st.spinner(f'{league_key}データをロード中...'):
            df = pd.read_csv(file_path)
            df['League'] = league_key
            if 'Match Date' in df.columns and 'Match ID' in df.columns:
                df['Match Date'] = pd.to_datetime(df['Match Date'], errors='coerce')
                unique_matches = df[['Team', 'Match ID', 'Match Date']].drop_duplicates().sort_values(by=['Team', 'Match Date'])
                unique_matches['Matchday'] = unique_matches.groupby('Team').cumcount() + 1
                df = pd.merge(df, unique_matches[['Team', 'Match ID', 'Matchday']], on=['Team', 'Match ID'], how='left')
                df = df.dropna(subset=['Matchday'])
                df['Matchday'] = df['Matchday'].astype(int)
            return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def get_all_league_data():
    all_dfs = [get_data(l) for l in LEAGUE_FILE_MAP.keys()]
    return pd.concat([d for d in all_dfs if not d.empty], ignore_index=True)

# --- 2. 定数定義 ---

LEAGUE_FILE_MAP = {'J1': '2025_J1_physical_data.csv', 'J2': '2025_J2_physical_data.csv', 'J3': '2025_J3_physical_data.csv'}
LEAGUE_COLOR_MAP = {'J1': '#E6002D', 'J2': '#127A3A', 'J3': '#014099'}

TEAM_COLORS = {
    'Kashima Antlers': '#B71940','Kashiwa Reysol':"#FFF000",'Urawa Red Diamonds': '#E6002D',
    'FC Tokyo': "#3E4C8D",'Tokyo Verdy':"#006931",'FC Machida Zelvia':"#0056A5",
    'Kawasaki Frontale': "#319FDA",'Yokohama F. Marinos': "#014099",'Yokohama FC':"#4BC1FE",'Shonan Bellmare':"#9EFF26",
    'Albirex Niigata':"#FE641E",'Shimizu S-Pulse':"#FF8901",'Nagoya Grampus': "#F8B500",
    'Kyoto Sanga FC':"#820064",'Gamba Osaka': "#00458D",'Cerezo Osaka': "#DB005B",'Vissel Kobe': '#A60129',
    'Fagiano Okayama':"#A72041",'Sanfrecce Hiroshima':"#603D97",'Avispa Fukuoka':"#9EB5C7",
    'Hokkaido Consadole Sapporo':"#125D75",'Vegalta Sendai':"#FFC20E",'AFC Blaublitz Akita':"#0D5790",'Montedio Yamagata':"#F7F4A6",'Iwaki SC':"#C01630",
    'Mito Hollyhock':"#2E3192",'Omiya Ardija':"#EC6601",'JEF United Ichihara Chiba':"#FFDE00",'Ventforet Kofu':"#0F63A3",
    'Kataller Toyama':"#25458F",'Jubilo Iwata':"#7294BA",'Fujieda MYFC':"#875884",'Renofa Yamaguchi':"#F26321",'Tokushima Vortis':"#11233F",'Ehime FC':"#ED9A4C",'FC Imabari':"#908E3C",
    'Sagan Tosu':"#30B7D7",'V-Varen Nagasaki':"#013893",'Roasso Kumamoto':"#A92D27",'Oita Trinita':"#254398",
    'Vanraure Hachinohe':"#13A63B",'Fukushima United FC':"#CF230C",'Tochigi SC':"#0170A4",'Tochigi City':"#001030",'ThespaKusatsu Gunma':"#08406F",'SC Sagamihara':"#408B52",
    'AC Parceiro Nagano':"#E36A2A",'Matsumoto Yamaga FC':"#004B1D",'Ishikawa FC Zweigen Kanazawa':"#3B1216",'FC Azul Claro Numazu':"#13A7DE",'FC Gifu':"#126246",
    'FC Osaka':"#90C9E2",'Nara Club':"#011D64",'Gainare Tottori':"#96C692",'Kamatamare Sanuki':"#669FB9",'Kochi United SC':"#B21E23",
    'Giravanz Kitakyushu':"#E8BD00",'Tegevajaro Miyazaki FC':"#F6E066",'Kagoshima United FC':"#19315F",'FC Ryūkyū':"#AA131B",
}

available_vars = [
    'Distance','Running Distance','M/min','HSR Distance','Sprint Count','HI Distance','HI Count',
    'Distance TIP','Running Distance TIP','HSR Distance TIP','HSR Count TIP',
    'Sprint Distance TIP','Sprint Count TIP','Distance OTIP','Running Distance OTIP',
    'HSR Distance OTIP','HSR Count OTIP','Sprint Distance OTIP','Sprint Count OTIP'
]
RANKING_METHODS = ['Total', 'Average', 'Max', 'Min']

# --- 3. 描画コンポーネント ---

def render_league_dashboard(df, league_name):
    """各リーグ共通のダッシュボード描画"""
    st.header(f"🏆 {league_name} リーグ分析")
    
    # チームカラーの準備
    current_teams = df['Team'].unique().tolist()
    domain = [t for t in current_teams if t in TEAM_COLORS]
    colors = [TEAM_COLORS[t] for t in domain]

    tab_rank, tab_custom, tab_trend = st.tabs(['集計ランキング', 'カスタムランキング', 'シーズン動向分析'])

    with tab_rank:
        st.markdown("### 📊 チーム別 ランキング")
        c1, c2 = st.columns(2)
        with c1: method = st.selectbox('集計方法', RANKING_METHODS, key=f'meth_{league_name}')
        with c2: 
            opts = [v.replace('Distance', 'Distance (km)') if v == 'Distance' and method == 'Total' else v for v in available_vars]
            selected_var = st.selectbox('表示指標', opts, key=f'var_{league_name}')
        
        actual_var = selected_var.replace(' (km)', '')

        # 💡 【重要】2段階集計ロジック
        # 1. 1試合ごとの合計を計算
        match_totals = df.groupby(['Team', 'Match ID'])[available_vars].sum().reset_index()
        
        # 2. 選択された方法で最終集計
        if method == 'Total': agg_df = match_totals.groupby('Team')[available_vars].sum().reset_index()
        elif method == 'Average': agg_df = match_totals.groupby('Team')[available_vars].mean().reset_index()
        elif method == 'Max': agg_df = match_totals.groupby('Team')[available_vars].max().reset_index()
        elif method == 'Min': agg_df = match_totals.groupby('Team')[available_vars].min().reset_index()

        if selected_var == 'Distance (km)':
            agg_df[selected_var] = agg_df['Distance'] / 1000
            plot_var = selected_var
        else:
            plot_var = actual_var

        sort_asc = (method == 'Min')
        
        chart = alt.Chart(agg_df).mark_bar().encode(
            y=alt.Y('Team:N', sort=alt.EncodingSortField(field=plot_var, order='ascending' if sort_asc else 'descending'), title='チーム'),
            x=alt.X(f'{plot_var}:Q', title=f'{method} {selected_var}'),
            color=alt.Color('Team:N', scale=alt.Scale(domain=domain, range=colors), legend=None),
            tooltip=['Team', alt.Tooltip(plot_var, format='.2f')]
        ).properties(height=600)
        
        st.altair_chart(chart, use_container_width=True)
        st.download_button("Excel出力", to_excel(agg_df[['Team', plot_var]]), f"{league_name}_{method}.xlsx")

    with tab_custom:
        # ご提示いただいたMatplotlibのランキング関数を呼び出し
        render_custom_ranking_core(df, league_name, TEAM_COLORS, available_vars)

    with tab_trend:
        # ご提示いただいた折れ線グラフの関数を呼び出し
        render_trend_analysis_core(df, league_name, TEAM_COLORS, available_vars)

# (内部用) カスタムランキングのコアロジック
def render_custom_ranking_core(df, league_name, team_colors, vars):
    st.markdown("### 🏆 カスタムランキング")
    team = st.selectbox('注目チームを選択', df['Team'].unique(), key=f"focal_{league_name}")
    col1, col2 = st.columns(2)
    with col1: method = st.selectbox('集計方法', RANKING_METHODS, key=f"cm_{league_name}")
    with col2: var = st.selectbox('評価指標', vars, key=f"cv_{league_name}")

    # 2段階集計適用
    match_totals = df.groupby(['Team', 'Match ID'])[vars].sum().reset_index()
    if method == 'Total': rank_df = match_totals.groupby('Team')[vars].sum().reset_index()
    elif method == 'Average': rank_df = match_totals.groupby('Team')[vars].mean().reset_index()
    elif method == 'Max': rank_df = match_totals.groupby('Team')[vars].max().reset_index()
    elif method == 'Min': rank_df = match_totals.groupby('Team')[vars].min().reset_index()

    sort_asc = (method == 'Min')
    sorted_df = rank_df.sort_values(by=var, ascending=sort_asc).reset_index(drop=True)
    if not sort_asc: sorted_df = sorted_df[::-1]

    # Matplotlib描画 (元のコードのスタイルを維持)
    fig, ax = plt.subplots(figsize=(7, 8), dpi=100)
    nrows = len(sorted_df)
    ax.set_xlim(0, 3); ax.set_ylim(0, nrows + 1.5)
    for i in range(nrows):
        t_name = sorted_df['Team'].iloc[i]
        val = sorted_df[var].iloc[i]
        is_focal = t_name == team
        rank = (nrows - i) if not sort_asc else (i + 1)
        color = team_colors.get(t_name, '#4A2E19') if is_focal else '#4A2E19'
        weight = 'bold' if is_focal else 'normal'
        ax.annotate(f"{rank}. {t_name}", xy=(0.1, i + 0.5), color=color, weight=weight)
        ax.annotate(f"{val/1000:.2f} km" if var == 'Distance' and method == 'Total' else f"{val:.2f}", xy=(2.2, i + 0.5), color=color, weight=weight)
    ax.set_axis_off()
    st.pyplot(fig)

# (内部用) シーズン動向のコアロジック
def render_trend_analysis_core(df, league_name, team_colors, vars):
    st.markdown("### 📈 シーズン動向分析")
    c1, c2 = st.columns(2)
    with c1: team = st.selectbox('チーム選択', sorted(df['Team'].unique()), key=f'tr_t_{league_name}')
    with c2: var = st.selectbox('指標選択', vars, key=f'tr_v_{league_name}')
    show_opp = st.checkbox('対戦相手を表示', key=f'opp_{league_name}')

    match_data = df.groupby(['Team', 'Match ID', 'Matchday'])[vars].sum().reset_index()
    team_data = match_data[match_data['Team'] == team]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=team_data['Matchday'], y=team_data[var], mode='lines+markers', name=team, line=dict(color=team_colors.get(team, '#4A2E19'))))
    
    if show_opp:
        opp_data = match_data[match_data['Match ID'].isin(team_data['Match ID']) & (match_data['Team'] != team)]
        fig.add_trace(go.Scatter(x=opp_data['Matchday'], y=opp_data[var], mode='lines+markers', name='Opponent', line=dict(dash='dot', color='gray')))
    
    fig.update_layout(xaxis_title='節', yaxis_title=var, height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- 4. メインロジック ---

with st.sidebar:
    selected = st.selectbox('リーグを選択', ['HOME', 'J1', 'J2', 'J3'], key='main_nav')

if selected == 'HOME':
    st.title('🇯🇵 J.League Data Dashboard')
    df_all = get_all_league_data()
    if not df_all.empty:
        col1, col2 = st.columns(2)
        with col1: x_v = st.selectbox('X軸', available_vars, index=1)
        with col2: y_v = st.selectbox('Y軸', available_vars, index=3)
        
        # 散布図はチーム平均で比較
        home_agg = df_all.groupby(['Team', 'League'])[available_vars].mean().reset_index()
        fig = px.scatter(home_agg, x=x_v, y=y_v, color='League', color_discrete_map=LEAGUE_COLOR_MAP, hover_data=['Team'], height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_all.head())
else:
    df_league = get_data(selected)
    if not df_league.empty:
        render_league_dashboard(df_league, selected)
