import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vega_datasets import data
import altair as alt

import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch, VerticalPitch

# --- 0. グローバル設定 ---
st.set_page_config(layout="wide")
st.subheader('All data by SkillCorner')

# --- 1. データと変数定義 (グローバルスコープ) ---
LEAGUE_FILE_MAP = {
    'J1': '2025_J1_physical_data.csv',
    'J2': '2025_J2_physical_data.csv', 
    'J3': '2025_J3_physical_data.csv', 
}
# リーグごとの指定色
LEAGUE_COLOR_MAP = {
    'J1': '#E6002D', # 赤
    'J2': '#127A3A', # 緑
    'J3': '#014099', # 青
}

@st.cache_data(ttl=60*15)
def get_data(league_key):
    file_name = LEAGUE_FILE_MAP.get(league_key, LEAGUE_FILE_MAP['J1'])
    file_path = f"data/{file_name}"
    try:
        # ローディングインジケータを表示 (Streamlit Cloudで役立つ)
        with st.spinner(f'{league_key}データをロード中...'):
            df = pd.read_csv(file_path)
            # リーグ情報を追加
            df['League'] = league_key
            return df
    except Exception as e:
        st.error(f"{league_key} データ ({file_name}) のロードに失敗しました。URLを確認してください: {file_path}")
        # st.exception(e) # デバッグ用
        return pd.DataFrame()

# 💡 新規: 全リーグデータを結合する関数
@st.cache_data(ttl=60*15)
def get_all_league_data():
    all_dfs = []
    for league_key in LEAGUE_FILE_MAP.keys():
        df = get_data(league_key)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

# 📌 チームカラー定義 (グローバルに配置)
TEAM_COLORS = {
    #J1 Teams
    'Kashima Antlers': '#B71940','Kashiwa Reysol':"#FFF000",'Urawa Red Diamonds': '#E6002D',
    'FC Tokyo': "#3E4C8D",'Tokyo Verdy':"#006931",'FC Machida Zelvia':"#0056A5",
    'Kawasaki Frontale': "#319FDA",'Yokohama F. Marinos': "#014099",'Yokohama FC':"#4BC1FE",'Shonan Bellmare':"#9EFF26",
    'Albirex Niigata':"#FE641E",'Shimizu S-Pulse':"#FF8901",'Nagoya Grampus': "#F8B500",
    'Kyoto Sanga FC':"#820064",'Gamba Osaka': "#00458D",'Cerezo Osaka': "#DB005B",'Vissel Kobe': '#A60129',
    'Fagiano Okayama':"#A72041",'Sanfrecce Hiroshima':"#603D97",'Avispa Fukuoka':"#9EB5C7",
    #J2 Teams
    'Hokkaido Consadole Sapporo':"#125D75",'Vegalta Sendai':"#FFC20E",'AFC Blaublitz Akita':"#0D5790",'Montedio Yamagata':"#F7F4A6",'Iwaki SC':"#C01630",
    'Mito Hollyhock':"#2E3192",'Omiya Ardija':"#EC6601",'JEF United Ichihara Chiba':"#FFDE00",'Ventforet Kofu':"#0F63A3",
    'Kataller Toyama':"#25458F",'Jubilo Iwata':"#7294BA",'Fujieda MYFC':"#875884",'Renofa Yamaguchi':"#F26321",'Tokushima Vortis':"#11233F",'Ehime FC':"#ED9A4C",'FC Imabari':"#908E3C",
    'Sagan Tosu':"#30B7D7",'V-Varen Nagasaki':"#013893",'Roasso Kumamoto':"#A92D27",'Oita Trinita':"#254398",
    #J3 Teams
    'Vanraure Hachinohe':"#13A63B",'Fukushima United FC':"#CF230C",
    'Tochigi SC':"#0170A4",'Tochigi City':"#001030",'ThespaKusatsu Gunma':"#08406F",'SC Sagamihara':"#408B52",
    'AC Parceiro Nagano':"#E36A2A",'Matsumoto Yamaga FC':"#004B1D",'Ishikawa FC Zweigen Kanazawa':"#3B1216",'FC Azul Claro Numazu':"#13A7DE",'FC Gifu':"#126246",
    'FC Osaka':"#90C9E2",'Nara Club':"#011D64",'Gainare Tottori':"#96C692",'Kamatamare Sanuki':"#669FB9",'Kochi United SC':"#B21E23",
    'Giravanz Kitakyushu':"#E8BD00",'Tegevajaro Miyazaki FC':"#F6E066",'Kagoshima United FC':"#19315F",'FC Ryūkyū':"#AA131B",
}

available_vars = ['Distance','Running Distance','HSR Distance','Sprint Count','HI Distance','HI Count',
                  'Distance TIP','Running Distance TIP','HSR Distance TIP','HSR Count TIP',]


# --- 2. 描画ロジック関数 (カスタムランキングを共通化) ---
# render_custom_ranking (変更なし)
# ...

# 💡 修正: Plotly Expressを使用した散布図描画関数
def render_scatter_plot(df: pd.DataFrame, available_vars: list, team_colors: dict, league_color_map: dict):
    """チーム別集計データに基づいて散布図を描画する"""
    st.markdown("### 📊 J.League 全体分析：散布図")
    st.markdown("チームごとの平均値を集計し、**2つの指標の関係性**を可視化します。")
    
    if 'League' not in df.columns:
        st.error("データに 'League' の列がありません。データロード関数を確認してください。")
        return
        
    team_avg_df = df.groupby(['Team', 'League'])[available_vars].mean().reset_index()

    if team_avg_df.empty:
        st.warning("集計データが空です。")
        return

    # UI要素の定義 (HOME全体でユニークなキーを設定)
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox('X軸の指標', available_vars, index=available_vars.index('Running Distance'), key='scatter_x_var_home')
    with col2:
        y_var = st.selectbox('Y軸の指標', available_vars, index=available_vars.index('HSR Distance'), key='scatter_y_var_home')
        
    # 🚨 修正点 1: 色分けの選択肢に「注目チーム」を追加
    color_by = st.radio('色分けの基準', ['リーグ', '注目チーム', 'なし'], index=0, key='scatter_color_by_home')
    
    # 注目チームの選択UI
    focal_team = None
    if color_by == '注目チーム':
        all_teams = sorted(team_avg_df['Team'].unique().tolist())

        default_index = all_teams.index('Yokohama FC') if 'Yokohama FC' in all_teams else 0
        focal_team = st.selectbox('注目チームを選択', all_teams, index=default_index, key='scatter_focal_team_home')

    # Plotly Expressで散布図を描画
    if color_by == 'リーグ':
        #リーグ色分けの指定を適用
        fig = px.scatter(
            team_avg_df, 
            x=x_var, 
            y=y_var, 
            color='League', 
            color_discrete_map=league_color_map, # 指定色を適用
            hover_data=['Team', 'League'],
            title=f'チーム別平均値: {y_var} vs {x_var} (リーグ別)',
            height=600,
        )
        
    elif color_by == '注目チーム' and focal_team:
        # 注目チームのデータフレームを作成
        team_avg_df['Highlight'] = team_avg_df['Team'].apply(
            lambda x: focal_team if x == focal_team else 'その他'
        )
        
        # 注目チームの色分けマップ
        highlight_color_map = {
            focal_team: team_colors.get(focal_team, '#FF0000'), # 注目チームの色
            'その他': '#CCCCCC' # それ以外のチームの色
        }

        fig = px.scatter(
            team_avg_df, 
            x=x_var, 
            y=y_var, 
            color='Highlight', 
            color_discrete_map=highlight_color_map,
            size='Distance', # サイズで総走行距離を表現（オプション）
            hover_data=['Team', 'League'],
            title=f'チーム別平均値: {y_var} vs {x_var} (注目チーム: {focal_team})',
            height=600,
        )
        
    else: # 'なし'またはフォールバックとしてチームカラーを使用 (前回の挙動を踏襲)
        all_team_colors = {team: team_colors.get(team, '#999999') for team in team_avg_df['Team'].unique()}
        
        fig = px.scatter(
            team_avg_df, 
            x=x_var, 
            y=y_var, 
            color='Team', 
            color_discrete_map=all_team_colors,
            hover_data=['Team', 'League'],
            title=f'チーム別平均値: {y_var} vs {x_var} (チーム別)',
            height=600,
        )


    # レイアウトの調整
    fig.update_layout(
        xaxis_title=f'{x_var} (平均)',
        yaxis_title=f'{y_var} (平均)',
        hovermode="closest",
    )
    
    # グラフを表示
    st.plotly_chart(fig, use_container_width=True)


# ... (render_custom_ranking 関数は省略/変更なし) ...


# --- 3. メインロジック ---

# サイドバーで選択と、その結果の変数 `selected` の取得のみを行う
with st.sidebar:
    st.subheader("menu")
    selected = st.selectbox(' ',['HOME','J1','J2','J3'], key='league_selector')
    
# サイドバーの外で、選択に基づきデータをロード
df = pd.DataFrame() 
if selected in ['J1', 'J2', 'J3']:
    df = get_data(selected) 
# 💡 変更: HOME選択時は全リーグデータをロード
elif selected == 'HOME':
    df = get_all_league_data()
else:
    df = pd.DataFrame() 

# --- 4. メインコンテンツの描画 ---

if selected == 'HOME':
    st.title('🇯🇵 J.League Data Dashboard: 全体分析')
    st.markdown('サイドバーからリーグを選択して、フィジカルデータ分析ダッシュボードをご利用ください。')
    
    if df.empty:
        st.warning("⚠️ J1, J2, J3 のいずれのデータもロードできなかったため、全体分析を表示できません。")
    else:
        # 💡 新規: 散布図タブを追加
        Scatter_tab, Preview_tab = st.tabs(['散布図分析', 'データプレビュー'])

        with Scatter_tab:
            # 🚨 修正: league_color_map を引数に追加
            render_scatter_plot(df, available_vars, TEAM_COLORS, LEAGUE_COLOR_MAP)

        with Preview_tab:
            st.subheader("全リーグデータプレビュー")
            st.dataframe(df.head())
            st.markdown(f"**ロードされたチーム数:** {df['Team'].nunique()} | **ロードされたデータ行数:** {len(df)}")

# ... (J1, J2, J3 のロジックは変更なし) ...
