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

st.subheader('All data by SkillCorner')

# --- 1. データと変数定義 (グローバルスコープ) ---
LEAGUE_FILE_MAP = {
    'J1': '2025_J1_physical_data.csv',
    'J2': '2025_J2_physical_data.csv', 
    'J3': '2025_J3_physical_data.csv', }

@st.cache_data(ttl=60*15)
def get_data(league_key):
    file_name = LEAGUE_FILE_MAP.get(league_key, LEAGUE_FILE_MAP['J1'])
    file_path = f"data/{file_name}"
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"{league_key} データ ({file_name}) のロードに失敗しました。URLを確認してください: {file_path}")
        return pd.DataFrame()

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
# --- 2. サイドバーとデータロードロジック ---

# サイドバーで選択と、その結果の変数 `selected` の取得のみを行う
with st.sidebar:
    st.subheader("menu")
    selected = st.selectbox(' ',['HOME','J1','J2','J3'], key='league_selector')
    
# サイドバーの外で、選択に基づきデータをロード
if selected in ['J1', 'J2', 'J3']:
    df = get_data(selected) 
elif selected == 'HOME':
    # HOMEではJ1をデフォルトでロード
    df = get_data('J1') 
else:
    df = pd.DataFrame() 

# --- 3. メインコンテンツの描画 ---

# 各セレクトボックス内容
if selected == 'HOME':
    st.dataframe(df.head())
    st.title('J.League Data Dashboard')
    st.markdown('サイドバーからリーグを選択して、フィジカルデータ分析ダッシュボードをご利用ください。')

# J1
if selected == 'J1':
    
    focal_color = '#000000'
    
    # データがロードされているか確認
    if df.empty:
        st.warning("データがロードされていないため、J1スタッツを表示できません。")
    else:
        
        # 🚨 修正 1: 表示対象のチームリストとカラー辞書を作成 🚨
        current_teams = df['Team'].unique().tolist()
        
        # TEAM_COLORSからcurrent_teamsに含まれる色のみを抽出
        filtered_colors = {team: TEAM_COLORS[team] for team in current_teams if team in TEAM_COLORS}

        # Altairで使用するためにキーと値のリストを作成
        domain_list = list(filtered_colors.keys())
        range_list = list(filtered_colors.values())
        
        # タブの作成
        Distance_tab, Sprint_table_tab, Test_tab = st.tabs(['総走行距離 (km)', '総スプリント数','Test'])
        
        # 💡 変数の初期化
        df_empty = pd.DataFrame()
        team_stats_aggregated = df_empty.copy()

        # チームごとの集計処理
        try:
            team_stats_aggregated = df.groupby('Team').agg(
                total_distance_m=('Distance', 'sum'),
                total_sprints=('Sprint Count', 'sum')
            ).reset_index()

            team_stats_aggregated['total_distance_km'] = team_stats_aggregated['total_distance_m'] / 1000
            team_stats_aggregated = team_stats_aggregated.set_index('Team')

            sorted_distance = team_stats_aggregated.sort_values(by='total_distance_km', ascending=False)
            sorted_sprints = team_stats_aggregated.sort_values(by='total_sprints', ascending=False)

            sorted_distance_reset = sorted_distance.reset_index()
            sorted_sprints_reset = sorted_sprints.reset_index()
            
            # --------------------
            # 🏃 総走行距離タブ (Altair描画)
            # --------------------
            with Distance_tab:
                st.markdown("### チーム別 総走行距離ランキング (km)")
                chart_distance = alt.Chart(sorted_distance_reset).mark_bar().encode(
                    y=alt.Y('Team:N', sort=alt.EncodingSortField(
                        field='total_distance_km', op='sum', order='descending'
                    ), title='チーム'),
                    x=alt.X('total_distance_km:Q', title='総走行距離 (km)'),
                    # 🚨 修正 2: フィルタリングされたカラーリストを使用 🚨
                    color=alt.Color('Team:N', scale=alt.Scale(domain=domain_list, range=range_list)),
                    tooltip=['Team', alt.Tooltip('total_distance_km', format='.1f')]
                ).properties(height=600)
                st.altair_chart(chart_distance, use_container_width=True)

            # --------------------
            # 💨 総スプリント数タブ (Altair描画)
            # --------------------
            with Sprint_table_tab:
                st.markdown("### チーム別 総スプリント数ランキング")
                
                chart_sprints = alt.Chart(sorted_sprints_reset).mark_bar().encode(
                    y=alt.Y('Team:N', sort=alt.EncodingSortField(
                        field='total_sprints', op='sum', order='descending'
                    ), title='チーム'),
                    x=alt.X('total_sprints:Q', title='総スプリント数'),
                    # 🚨 修正 3: フィルタリングされたカラーリストを使用 🚨
                    color=alt.Color('Team:N', scale=alt.Scale(domain=domain_list, range=range_list)),
                    tooltip=['Team', 'total_sprints']
                ).properties(height=600)
                st.altair_chart(chart_sprints, use_container_width=True)

        except KeyError as e:
            st.error(f"データ集計に失敗しました。CSVファイルに必須の列が見つかりません: {e}")
        except Exception as e:
            st.error(f"予期せぬエラーが発生しました: {e}")

        #---------
        with Test_tab:
            st.markdown("### 🏆 カスタムランキング作成")
            
            # データの準備 (Test_tab 内で定義)
            ranking_base_df = df.copy()
            
            # UI要素の定義 (正しいインデント)
            team = st.selectbox('注目チームを選択', df['Team'].unique(), key="focal_team_J1") 
            focal_color = TEAM_COLORS.get(team, '#000000') 

            col1, col2 = st.columns(2)
            with col1:
                rank_method = st.selectbox('集計方法 (Ranking Method)', ['Average', 'Total', 'Max', 'Min'], key="rank_method_J1") 
            with col2:
                rank_var = st.selectbox('評価指標 (Metric to Rank)', available_vars, key="rank_var_J1") 
            
            # データの集計ロジック (with Test_tab: の正しいインデント内)
            if rank_method == 'Total':
                rank_df = ranking_base_df.groupby(['Team'])[available_vars].sum().reset_index()
            elif rank_method == 'Average':
                rank_df = ranking_base_df.groupby(['Team'])[available_vars].mean().reset_index()
            elif rank_method == 'Max':
                rank_df = ranking_base_df.groupby(['Team'])[available_vars].max().reset_index()
            elif rank_method == 'Min':
                rank_df = ranking_base_df.groupby(['Team'])[available_vars].min().reset_index()

            # 最終的なランキングデータフレームの作成
            sort_method = False
            indexdf_short = rank_df.sort_values(by=[rank_var],ascending=sort_method)[['Team',rank_var]].reset_index(drop=True)[::-1]
            
            # データが空の場合のチェック (インデント)
            if indexdf_short.empty:
                st.warning("集計されたデータが空のため、ランキングを表示できません。")
            else:
                # Matplotlib/Seabornの設定 (インデント)
                sns.set(rc={'axes.facecolor':'#fbf9f4', 'figure.facecolor':'#fbf9f4',
                            'ytick.labelcolor':'#4A2E19', 'xtick.labelcolor':'#4A2E19'})

                fig = plt.figure(figsize=(7, 8), dpi=200)
                ax = plt.subplot()
                
                # 描画に必要な値の計算
                ncols = len(indexdf_short.columns.tolist()) + 1
                nrows = indexdf_short.shape[0]

                ax.set_xlim(0, ncols + .5)
                ax.set_ylim(0, nrows + 1.5)
                
                positions = [0.05, 2.0] # チーム名と指標値のX座標
                columns = indexdf_short.columns.tolist()
                
                # テーブルのメインテキスト描画
                for i in range(nrows):
                    for j, column in enumerate(columns):
                        if column == 'Team':
                            rank = nrows - i
                            text_label = f'{rank}     {indexdf_short[column].iloc[i]}' if rank < 10 else f'{rank}   {indexdf_short[column].iloc[i]}'
                        else:
                            text_label = f'{round(indexdf_short[column].iloc[i],2)}'
                        
                        t_color = focal_color if indexdf_short['Team'].iloc[i] == team else '#4A2E19'
                        weight = 'bold' if indexdf_short['Team'].iloc[i] == team else 'regular'
                        
                        ax.annotate(
                            xy=(positions[j], i + .5),
                            text = text_label,
                            ha='left', va='center', color=t_color, weight=weight
                        )
                        
                # テーブルヘッダー描画
                column_names = ['Rank / Team', rank_var]
                for index, cs in enumerate(column_names):
                        pos = positions[index]
                        ax.annotate(
                            xy=(pos, nrows + .75),
                            text=column_names[index],
                            ha='left', va='bottom', weight='bold', color='#4A2E19'
                        )

                # 罫線
                ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows + 0.5, nrows + 0.5], lw=1.5, color='black', marker='', zorder=4)
                ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
                for x in range(1, nrows):
                    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')
                
                ax.set_axis_off() 
                
                # タイトル描画
                fig.text(
                    x=0.08, y=.95, s=f"{rank_var} {rank_method} Rankings",
                    ha='left', va='bottom', weight='bold', size=13, color='#4A2E19')
                
                # Streamlitで図を表示
                st.pyplot(fig)