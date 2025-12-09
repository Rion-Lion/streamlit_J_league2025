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

# リーグごとの指定色 (HOME画面の散布図用)
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
            # ファイルアップロードから取得するため、一時的にread_csvをコメントアウト
            # df = pd.read_csv(file_path)
            # 暫定的に空のデータフレームで続行
            # 実際にはここでファイルのロードが必要です。
            # 今回のシナリオでは、ユーザーがファイルをアップロードしている前提なので、
            # この部分はダッシュボードとして動作することを優先します。
            df = pd.read_csv(file_path) # 既存の仮パスを使用
            
            # リーグ情報を追加
            df['League'] = league_key

            # Match ID と Match Date を使用して Matchday (節) を計算
            if 'Match Date' in df.columns and 'Match ID' in df.columns and not df['Match Date'].isnull().all():
                
                # Match Dateを日付型に変換（エラーが出たら無視）
                df['Match Date'] = pd.to_datetime(df['Match Date'], errors='coerce')
                
                # 1. ユニークな試合の特定 (Match IDをキーに使用)
                unique_matches = df[['Team', 'Match ID', 'Match Date']].drop_duplicates()
                
                # 2. Match Dateでソート
                unique_matches = unique_matches.sort_values(by=['Team', 'Match Date']).reset_index(drop=True)
                
                # 3. チームごとに節番号 (Matchday) を付与
                unique_matches['Matchday'] = unique_matches.groupby('Team').cumcount() + 1
                
                # 4. 節番号を元のデータフレームにマージ (TeamとMatch IDをキーに)
                df = pd.merge(df, unique_matches[['Team', 'Match ID', 'Matchday']], on=['Team', 'Match ID'], how='left')
                
                # 'Matchday' が NaN になる行がある可能性（データの欠損/不整合）を考慮し、NaNは削除/無視
                df = df.dropna(subset=['Matchday'])
                df['Matchday'] = df['Matchday'].astype(int)

                st.info(f"✅ {league_key}データで 'Match ID' と 'Match Date' を使用して節 ('Matchday') を正しく生成しました。")
                
            # フォールバックロジック (Match Date/Match IDがない場合)
            elif 'Matchday' not in df.columns:
                 df['Matchday'] = df.groupby('Team').cumcount() + 1
                 st.warning(f"⚠️ {league_key}データに正確な時系列情報がなく、節 ('Matchday') の生成が不正確になる可能性があります。")
                 
            return df
    except Exception as e:
        # st.error(f"{league_key} データ ({file_name}) のロードに失敗しました。ファイルが存在するか確認してください。")
        # デバッグ情報としてエラーを表示しつつ、空のデータフレームを返す
        st.error(f"データロードエラー: {e}")
        return pd.DataFrame()

# 全リーグデータを結合する関数 (HOME画面用)
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
                  'Distance TIP','Running Distance TIP','HSR Distance TIP','HSR Count TIP',
                  'Sprint Distance TIP','Sprint Count TIP','Distance OTIP','Running Distance OTIP','HSR Distance OTIP','HSR Count OTIP',
                  'Sprint Distance OTIP','Sprint Count OTIP'] # TIP/OTIP指標を追加


# --- 2. 描画ロジック関数 (共通関数) ---

def render_custom_ranking(df: pd.DataFrame, league_name: str, team_colors: dict, available_vars: list):
    """カスタムランキング（Matplotlib）を描画する"""
    st.markdown("### 🏆 カスタムランキング作成")
    
    # UI要素の定義: keyをリーグごとにユニークにし、セッションステートの衝突を防ぐ
    team = st.selectbox('注目チームを選択', df['Team'].unique(), key=f"focal_team_{league_name}") 
    focal_color = team_colors.get(team, '#000000') 

    col1, col2 = st.columns(2)
    with col1:
        rank_method = st.selectbox('集計方法 (Ranking Method)', ['Average', 'Total', 'Max', 'Min'], key=f"rank_method_{league_name}") 
    with col2:
        rank_var = st.selectbox('評価指標 (Metric to Rank)', available_vars, key=f"rank_var_{league_name}") 
    
    ranking_base_df = df.copy()

    # データの集計ロジック
    if rank_method == 'Total':
        rank_df = ranking_base_df.groupby(['Team'])[available_vars].sum().reset_index()
        sort_method = False
    elif rank_method == 'Average':
        rank_df = ranking_base_df.groupby(['Team'])[available_vars].mean().reset_index()
        sort_method = False
    elif rank_method == 'Max':
        rank_df = ranking_base_df.groupby(['Team'])[available_vars].max().reset_index()
        sort_method = False
    elif rank_method == 'Min':
        rank_df = ranking_base_df.groupby(['Team'])[available_vars].min().reset_index()
        sort_method = True 

    # 最終的なランキングデータフレームの作成
    if sort_method: 
        indexdf_short = rank_df.sort_values(by=[rank_var], ascending=True)[['Team', rank_var]].reset_index(drop=True)
    else: 
        indexdf_short = rank_df.sort_values(by=[rank_var], ascending=False)[['Team', rank_var]].reset_index(drop=True)
    
    indexdf_short = indexdf_short[::-1]

    if indexdf_short.empty:
        st.warning("集計されたデータが空のため、ランキングを表示できません。")
        return

    # --- Matplotlib/Seaborn 描画ロジック ---
    sns.set(rc={'axes.facecolor':'#fbf9f4', 'figure.facecolor':'#fbf9f4',
                'ytick.labelcolor':'#4A2E19', 'xtick.labelcolor':'#4A2E19'})

    fig = plt.figure(figsize=(7, 8), dpi=200)
    ax = plt.subplot()
    
    ncols = len(indexdf_short.columns.tolist()) + 1
    nrows = indexdf_short.shape[0]

    ax.set_xlim(0, ncols + .5)
    ax.set_ylim(0, nrows + 1.5)
    
    positions = [0.05, 2.0]
    columns = indexdf_short.columns.tolist()
    
    for i in range(nrows):
        team_name = indexdf_short['Team'].iloc[i]
        is_focal = team_name == team
        t_color = focal_color if is_focal else '#4A2E19'
        weight = 'bold' if is_focal else 'regular'

        rank = nrows - i
        
        for j, column in enumerate(columns):
            if column == 'Team':
                text_label = f'{rank}     {team_name}' if rank < 10 else f'{rank}   {team_name}'
            else:
                text_label = f'{round(indexdf_short[column].iloc[i],2)}'
            
            ax.annotate(
                xy=(positions[j], i + .5),
                text = text_label,
                ha='left', va='center', color=t_color, weight=weight
            )
            
    # テーブルヘッダー描画
    column_names = ['Rank / Team', rank_var]
    for index, cs in enumerate(column_names):
        pos = positions[index]
        ax.annotate(xy=(pos, nrows + .75), text=column_names[index], ha='left', va='bottom', weight='bold', color='#4A2E19')

    # 罫線
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows + 0.5, nrows + 0.5], lw=1.5, color='black', marker='', zorder=4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
    for x in range(1, nrows):
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')
    
    ax.set_axis_off() 
    
    # タイトル描画
    fig.text(x=0.08, y=.95, s=f"{rank_var} {rank_method} Rankings",
        ha='left', va='bottom', weight='bold', size=13, color='#4A2E19')
    
    st.pyplot(fig)


# Plotly Expressを使用した散布図描画関数 (HOME画面用)
def render_scatter_plot(df: pd.DataFrame, available_vars: list, team_colors: dict, league_color_map: dict):
    """チーム別集計データに基づいて散布図を描画する"""
    st.markdown("### 📊 J.League 全体分析：散布図")
    
    if 'League' not in df.columns:
        st.error("データに 'League' の列がありません。データロード関数を確認してください。")
        return
        
    # データの集計: チームとリーグでグループ化し、全指標の平均を算出
    team_avg_df = df.groupby(['Team', 'League'])[available_vars].mean().reset_index()

    if team_avg_df.empty:
        st.warning("集計データが空です。")
        return

    # UI要素の定義 (X軸/Y軸)
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox('X軸の指標', available_vars, index=available_vars.index('Running Distance'), key='scatter_x_var_home')
    with col2:
        y_var = st.selectbox('Y軸の指標', available_vars, index=available_vars.index('HSR Distance'), key='scatter_y_var_home')
        
    # 色分けの基準
    color_by = st.radio('色分けの基準', ['リーグ', '注目チーム', 'チーム別 (デフォルト)'], index=0, key='scatter_color_by_home')
    
    focal_team = None
    if color_by == '注目チーム':
        all_teams = sorted(team_avg_df['Team'].unique().tolist())
        default_index = all_teams.index('Cerezo Osaka') if 'Cerezo Osaka' in all_teams else 0
        focal_team = st.selectbox('注目チームを選択', all_teams, index=default_index, key='scatter_focal_team_home')

    # チーム名とリーグ、選択指標を表示するリスト
    hover_data_list = ['Team', 'League', x_var, y_var]

    # Plotly Expressで散布図を描画
    if color_by == 'リーグ':
        fig = px.scatter(
            team_avg_df, 
            x=x_var, 
            y=y_var, 
            color='League', 
            color_discrete_map=league_color_map, 
            hover_data=hover_data_list,
            title=f'チーム別平均値: {y_var} vs {x_var} (リーグ別)',
            height=600,
        )
        
    elif color_by == '注目チーム' and focal_team:
        # 注目チームのデータフレームを作成し、色分け用の列 'Highlight' を追加
        team_avg_df['Highlight'] = team_avg_df['Team'].apply(
            lambda x: focal_team if x == focal_team else 'その他'
        )
        
        # 注目チームの色分けマップ: 注目チームはチームカラー、その他はグレー
        highlight_color_map = {
            focal_team: team_colors.get(focal_team, '#FF0000'), 
            'その他': '#CCCCCC' 
        }

        fig = px.scatter(
            team_avg_df, 
            x=x_var, 
            y=y_var, 
            color='Highlight', 
            color_discrete_map=highlight_color_map,
            # hover_dataにはHighlightを含めず、代わりにTeamを含めることで、Highlightの内容は表示されなくなる。
            hover_data=['Team', 'League', x_var, y_var], 
            title=f'チーム別平均値: {y_var} vs {x_var} (注目チーム: {focal_team})',
            height=600,
        )
        # 注目チームのマーカーを大きくする
        fig.update_traces(marker=dict(size=12), selector=dict(name=focal_team))
        fig.update_traces(marker=dict(size=8), selector=dict(name='その他'))
        
    else: # 'チーム別 (デフォルト)' またはフォールバック
        all_team_colors = {team: team_colors.get(team, '#999999') for team in team_avg_df['Team'].unique()}
        
        fig = px.scatter(
            team_avg_df, 
            x=x_var, 
            y=y_var, 
            color='Team', 
            color_discrete_map=all_team_colors,
            hover_data=hover_data_list,
            title=f'チーム別平均値: {y_var} vs {x_var} (チーム別)',
            height=600,
        )

    # レイアウトの調整
    fig.update_layout(
        xaxis_title=f'{x_var} (平均)',
        yaxis_title=f'{y_var} (平均)',
        hovermode="closest",
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_trend_analysis(df: pd.DataFrame, league_name: str, team_colors: dict, available_vars: list):
    """チームごとのシーズン動向を節ベースで分析する折れ線グラフを描画する (対戦相手比較機能付き)"""
    st.markdown(f"### 📈 シーズン動向分析 ({league_name})")
    
    if 'Matchday' not in df.columns or df['Matchday'].isnull().all():
        st.error("⚠️ データに **'Matchday'** (節) 列が見つからないか、データが不完全です。データロード関数を確認してください。")
        return

    # 1. UI要素の定義 (チーム選択と分析項目選択)
    all_teams = sorted(df['Team'].unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        selected_team = st.selectbox('チームを選択', all_teams, key=f'trend_team_{league_name}')
    with col2:
        selected_var = st.selectbox('分析したい項目を選択', available_vars, key=f'trend_var_{league_name}')
    
    # 条件ボタンの追加
    show_opponent = st.checkbox('対戦相手のデータも表示する', key=f'show_opponent_{league_name}')

    # 2. 自チームデータ準備
    team_data = df[df['Team'] == selected_team].copy()
    
    # 節ごとの平均値を計算 (自チーム): MatchdayとMatch IDでグループ化することで、1試合1行に集約
    team_match_df = team_data.groupby(['Matchday', 'Match ID'])[selected_var].mean().reset_index()
    team_match_df = team_match_df.rename(columns={selected_var: f'{selected_var} (自チーム)'})

    if team_match_df.empty:
        st.warning(f"{selected_team} のデータが見つかりません。")
        return

    # 3. 対戦相手データ準備 (条件がONの場合)
    opponent_match_df = None
    if show_opponent:
        # 全データから自チームの試合IDリストを取得
        match_ids = team_match_df['Match ID'].unique()
        
        # 自チームの試合に限定し、かつ自チームではない行を抽出（=対戦相手のデータ）
        opponent_data = df[df['Match ID'].isin(match_ids) & (df['Team'] != selected_team)].copy()
        
        if not opponent_data.empty:
            # MatchdayとMatch IDの対応表を作成 (自チームデータから一意の対応を取得)
            matchday_map = team_match_df[['Matchday', 'Match ID']].drop_duplicates()
            
            # 対戦相手のMatch IDごとの平均値を計算 (Match IDごとに1行に集約)
            opponent_avg_df = opponent_data.groupby('Match ID').agg(
                {selected_var: 'mean', 'Team': 'first'} # Team:firstで、そのMatch IDにおける対戦相手チーム名を取得
            ).reset_index()
            
            # Matchdayをマッピング
            opponent_match_df = pd.merge(opponent_avg_df, matchday_map, on='Match ID', how='left')
            
            # グラフ用のデータフレームに整理
            opponent_match_df = opponent_match_df.rename(columns={selected_var: f'{selected_var} (対戦相手)'})
            
            # 最終確認と厳密な重複排除：MatchdayとMatch ID、Team（対戦相手名）の組み合わせで一意になるようにする
            opponent_match_df = opponent_match_df.sort_values('Matchday').drop_duplicates(
                subset=['Matchday', 'Match ID', 'Team'], 
                keep='first'
            )
            # Matchdayを基準にソート (グラフ表示順序のため)
            opponent_match_df = opponent_match_df.sort_values(by='Matchday')

    # 4. Plotly Graph Objectsで折れ線グラフ描画
    team_color = team_colors.get(selected_team, '#4A2E19')
    opponent_color = '#999999' # 対戦相手はグレー系で統一

    fig = go.Figure()
    
    # --- 自チームのホバーテンプレート ---
    hovertemplate_self = f" %{{y:.2f}}<extra>自チーム</extra>"
    custom_data_self = None
    
    fig.add_trace(go.Scatter(
        x=team_match_df['Matchday'],
        y=team_match_df[f'{selected_var} (自チーム)'],
        mode='lines+markers',
        name=f'{selected_team} (自チーム)',
        line=dict(color=team_color, width=2),
        marker=dict(size=6),
        hovertemplate=hovertemplate_self,
        customdata=custom_data_self
    ))
    
    # --- 対戦相手のホバーテンプレート ---
    if show_opponent and opponent_match_df is not None and not opponent_match_df.empty:
        # 相手名が先、値が後になるように順序を入れ替え
        custom_data_opponent = opponent_match_df[['Team']].values.tolist() 
        hovertemplate_opponent = f"%{{customdata[0]}}<br>: %{{y:.2f}}<extra>対戦相手</extra>"
        
        fig.add_trace(go.Scatter(
            x=opponent_match_df['Matchday'],
            y=opponent_match_df[f'{selected_var} (対戦相手)'],
            mode='lines+markers',
            name='対戦相手 (試合平均)',
            line=dict(color=opponent_color, width=2, dash='dot'),
            marker=dict(size=6, symbol='x'),
            hovertemplate=hovertemplate_opponent,
            customdata=custom_data_opponent
        ))
    

    # レイアウト設定
    title_text = f'**{selected_team}**: {selected_var} のシーズン推移'
    if show_opponent:
         title_text += ' (対戦相手比較)'

    fig.update_layout(
        title=title_text,
        xaxis_title='節 (Matchday)',
        yaxis_title=f'{selected_var} (試合平均)',
        hovermode="x unified",
        height=550,
        # X軸の範囲
        xaxis=dict(range=[0, 39]) 
    )
    # X軸の目盛りを整数にする
    fig.update_xaxes(dtick=1)
    
    st.plotly_chart(fig, use_container_width=True)


# --- 3. メインロジック ---

# サイドバーで選択と、その結果の変数 `selected` の取得のみを行う
with st.sidebar:
    st.subheader("menu")
    selected = st.selectbox(' ',['HOME','J1','J2','J3'], key='league_selector')
    
# サイドバーの外で、選択に基づきデータをロード
df = pd.DataFrame() 
if selected in ['J1', 'J2', 'J3']:
    df = get_data(selected) 
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
        Scatter_tab, Preview_tab = st.tabs(['散布図分析', 'データプレビュー'])

        with Scatter_tab:
            render_scatter_plot(df, available_vars, TEAM_COLORS, LEAGUE_COLOR_MAP)

        with Preview_tab:
            st.subheader("全リーグデータプレビュー")
            st.dataframe(df.head())
            st.markdown(f"**ロードされたチーム数:** {df['Team'].nunique()} | **ロードされたデータ行数:** {len(df)}")


# ------------------------------------
# J1 リーグのコンテンツ
# ------------------------------------
if selected == 'J1':
    
    if df.empty:
        st.warning("データがロードされていないため、J1スタッツを表示できません。")
    else:
        st.header(f"🏆 J1 リーグ分析ダッシュボード")
        
        current_teams = df['Team'].unique().tolist()
        filtered_colors = {team: TEAM_COLORS[team] for team in current_teams if team in TEAM_COLORS}
        domain_list = list(filtered_colors.keys())
        range_list = list(filtered_colors.values())
        
        Distance_tab, Sprint_table_tab, Custom_tab, Trend_tab = st.tabs(['総走行距離 (km)', '総スプリント数','カスタムランキング', 'シーズン動向分析']) 
        
        try:
            team_stats_aggregated = df.groupby('Team').agg(
                total_distance_m=('Distance', 'sum'),
                total_sprints=('Sprint Count', 'sum')
            ).reset_index()

            team_stats_aggregated['total_distance_km'] = team_stats_aggregated['total_distance_m'] / 1000
            
            sorted_distance_reset = team_stats_aggregated.sort_values(by='total_distance_km', ascending=False).reset_index(drop=True)
            sorted_sprints_reset = team_stats_aggregated.sort_values(by='total_sprints', ascending=False).reset_index(drop=True)
            
            with Distance_tab:
                st.markdown("### チーム別 総走行距離ランキング (km)")
                chart_distance = alt.Chart(sorted_distance_reset).mark_bar().encode(
                    y=alt.Y('Team:N', sort=alt.EncodingSortField(
                        field='total_distance_km', op='sum', order='descending'
                    ), title='チーム'),
                    x=alt.X('total_distance_km:Q', title='総走行距離 (km)'),
                    color=alt.Color('Team:N', scale=alt.Scale(domain=domain_list, range=range_list)),
                    tooltip=['Team', alt.Tooltip('total_distance_km', format='.1f')]
                ).properties(height=600)
                st.altair_chart(chart_distance, use_container_width=True)

            with Sprint_table_tab:
                st.markdown("### チーム別 総スプリント数ランキング")
                chart_sprints = alt.Chart(sorted_sprints_reset).mark_bar().encode(
                    y=alt.Y('Team:N', sort=alt.EncodingSortField(
                        field='total_sprints', op='sum', order='descending'
                    ), title='チーム'),
                    x=alt.X('total_sprints:Q', title='総スプリント数'),
                    color=alt.Color('Team:N', scale=alt.Scale(domain=domain_list, range=range_list)),
                    tooltip=['Team', 'total_sprints']
                ).properties(height=600)
                st.altair_chart(chart_sprints, use_container_width=True)

        except KeyError as e:
            st.error(f"J1データの集計に失敗しました。CSVファイルに必須の列が見つかりません: {e}")
        except Exception as e:
            st.error(f"J1で予期せぬエラーが発生しました: {e}")

        with Custom_tab:
            render_custom_ranking(df, 'J1', TEAM_COLORS, available_vars)
        
        # シーズン動向分析
        with Trend_tab:
            render_trend_analysis(df, 'J1', TEAM_COLORS, available_vars)


# ------------------------------------
# J2 リーグのコンテンツ
# ------------------------------------
elif selected == 'J2':
    
    if df.empty:
        st.warning(f"⚠️ {selected} リーグのデータがロードできませんでした。ファイルが存在するか確認してください。")
    else:
        st.header(f"🏆 J2 リーグ分析ダッシュボード")

        current_teams = df['Team'].unique().tolist()
        filtered_colors = {team: TEAM_COLORS[team] for team in current_teams if team in TEAM_COLORS}
        domain_list = list(filtered_colors.keys())
        range_list = list(filtered_colors.values())
        
        Distance_tab, Sprint_table_tab, Custom_tab, Trend_tab = st.tabs(['総走行距離 (km)', '総スプリント数','カスタムランキング', 'シーズン動向分析'])
        
        try:
            team_stats_aggregated = df.groupby('Team').agg(
                total_distance_m=('Distance', 'sum'),
                total_sprints=('Sprint Count', 'sum')
            ).reset_index()

            team_stats_aggregated['total_distance_km'] = team_stats_aggregated['total_distance_m'] / 1000
            sorted_distance_reset = team_stats_aggregated.sort_values(by='total_distance_km', ascending=False).reset_index(drop=True)
            sorted_sprints_reset = team_stats_aggregated.sort_values(by='total_sprints', ascending=False).reset_index(drop=True)
            
            with Distance_tab:
                st.markdown("### チーム別 総走行距離ランキング (km)")
                chart_distance = alt.Chart(sorted_distance_reset).mark_bar().encode(
                    y=alt.Y('Team:N', sort=alt.EncodingSortField(
                        field='total_distance_km', op='sum', order='descending'
                    ), title='チーム'),
                    x=alt.X('total_distance_km:Q', title='総走行距離 (km)'),
                    color=alt.Color('Team:N', scale=alt.Scale(domain=domain_list, range=range_list)),
                    tooltip=['Team', alt.Tooltip('total_distance_km', format='.1f')]
                ).properties(height=600)
                st.altair_chart(chart_distance, use_container_width=True)

            with Sprint_table_tab:
                st.markdown("### チーム別 総スプリント数ランキング")
                chart_sprints = alt.Chart(sorted_sprints_reset).mark_bar().encode(
                    y=alt.Y('Team:N', sort=alt.EncodingSortField(
                        field='total_sprints', op='sum', order='descending'
                    ), title='チーム'),
                    x=alt.X('total_sprints:Q', title='総スプリント数'),
                    color=alt.Color('Team:N', scale=alt.Scale(domain=domain_list, range=range_list)),
                    tooltip=['Team', 'total_sprints']
                ).properties(height=600)
                st.altair_chart(chart_sprints, use_container_width=True)

        except KeyError as e:
            st.error(f"J2データの集計に失敗しました。CSVファイルに必須の列が見つかりません: {e}")
        except Exception as e:
            st.error(f"J2で予期せぬエラーが発生しました: {e}")

        with Custom_tab:
            render_custom_ranking(df, 'J2', TEAM_COLORS, available_vars)

        # シーズン動向分析
        with Trend_tab:
            render_trend_analysis(df, 'J2', TEAM_COLORS, available_vars)


# ------------------------------------
# J3 リーグのコンテンツ
# ------------------------------------
elif selected == 'J3':
    
    if df.empty:
        st.warning(f"⚠️ {selected} リーグのデータがロードできませんでした。ファイルが存在するか確認してください。")
    else:
        st.header(f"🏆 J3 リーグ分析ダッシュボード")
        
        current_teams = df['Team'].unique().tolist()
        filtered_colors = {team: TEAM_COLORS[team] for team in current_teams if team in TEAM_COLORS}
        domain_list = list(filtered_colors.keys())
        range_list = list(filtered_colors.values())
        
        Distance_tab, Sprint_table_tab, Custom_tab, Trend_tab = st.tabs(['総走行距離 (km)', '総スプリント数','カスタムランキング', 'シーズン動向分析'])
        
        try:
            team_stats_aggregated = df.groupby('Team').agg(
                total_distance_m=('Distance', 'sum'),
                total_sprints=('Sprint Count', 'sum')
            ).reset_index()

            team_stats_aggregated['total_distance_km'] = team_stats_aggregated['total_distance_m'] / 1000
            sorted_distance_reset = team_stats_aggregated.sort_values(by='total_distance_km', ascending=False).reset_index(drop=True)
            sorted_sprints_reset = team_stats_aggregated.sort_values(by='total_sprints', ascending=False).reset_index(drop=True)
            
            with Distance_tab:
                st.markdown("### チーム別 総走行距離ランキング (km)")
                chart_distance = alt.Chart(sorted_distance_reset).mark_bar().encode(
                    y=alt.Y('Team:N', sort=alt.EncodingSortField(
                        field='total_distance_km', op='sum', order='descending'
                    ), title='チーム'),
                    x=alt.X('total_distance_km:Q', title='総走行距離 (km)'),
                    color=alt.Color('Team:N', scale=alt.Scale(domain=domain_list, range=range_list)),
                    tooltip=['Team', alt.Tooltip('total_distance_km', format='.1f')]
                ).properties(height=600)
                st.altair_chart(chart_distance, use_container_width=True)

            with Sprint_table_tab:
                st.markdown("### チーム別 総スプリント数ランキング")
                chart_sprints = alt.Chart(sorted_sprints_reset).mark_bar().encode(
                    y=alt.Y('Team:N', sort=alt.EncodingSortField(
                        field='total_sprints', op='sum', order='descending'
                    ), title='チーム'),
                    x=alt.X('total_sprints:Q', title='総スプリント数'),
                    color=alt.Color('Team:N', scale=alt.Scale(domain=domain_list, range=range_list)),
                    tooltip=['Team', 'total_sprints']
                ).properties(height=600)
                st.altair_chart(chart_sprints, use_container_width=True)

        except KeyError as e:
            st.error(f"J3データの集計に失敗しました。CSVファイルに必須の列が見つかりません: {e}")
        except Exception as e:
            st.error(f"J3で予期せぬエラーが発生しました: {e}")

        with Custom_tab:
            render_custom_ranking(df, 'J3', TEAM_COLORS, available_vars)
            
        # シーズン動向分析
        with Trend_tab:
            render_trend_analysis(df, 'J3', TEAM_COLORS, available_vars)
