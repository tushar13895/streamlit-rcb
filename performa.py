#!/usr/bin/env python
# coding: utf-8



from pandas.core.base import DataError
from datetime  import date
from datetime import timedelta
from datetime import datetime
from plotly import graph_objs as go

import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot  as plt

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

from sklearn.cluster import DBSCAN

import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import streamlit as st
import time
import base64

#Set the page content to cover entire screen
st.set_page_config(layout="wide")



#Function to load the title page
def title():
    st.image("RCBSquadWeb.jpeg", use_column_width='always')

def batsman():
    html_temp = "<div class='tableauPlaceholder' id='viz1655047148907' style='position: relative'><noscript><a href='#'><img alt='Batting Analysis ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;IP&#47;IPLDescriptiveAnalysis_16550346868690&#47;BattingAnalysis&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IPLDescriptiveAnalysis_16550346868690&#47;BattingAnalysis' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;IP&#47;IPLDescriptiveAnalysis_16550346868690&#47;BattingAnalysis&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1655047148907');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1200px';vizElement.style.height='1227px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1200px';vizElement.style.height='1227px';} else { vizElement.style.width='100%';vizElement.style.height='1877px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
    components.html(html_temp, width=1300, height=550, scrolling=True)
    
def bowler():
    html_temp = "<div class='tableauPlaceholder' id='viz1655050163013' style='position: relative'><noscript><a href='#'><img alt='Bowling Analysis ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;IP&#47;IPLDescriptiveAnalysis_16550346868690&#47;BowlingAnalysis&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IPLDescriptiveAnalysis_16550346868690&#47;BowlingAnalysis' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;IP&#47;IPLDescriptiveAnalysis_16550346868690&#47;BowlingAnalysis&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1655050163013');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1200px';vizElement.style.height='1227px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1200px';vizElement.style.height='1227px';} else { vizElement.style.width='100%';vizElement.style.height='1527px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
    components.html(html_temp, width=1300, height=550, scrolling=True)
    
def team():
    html_temp = "<div class='tableauPlaceholder' id='viz1655122291257' style='position: relative'><noscript><a href='#'><img alt='Team and Venue Analysis ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;IP&#47;IPLDescriptiveAnalysis_16550346868690&#47;TeamandVenueAnalysis&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IPLDescriptiveAnalysis_16550346868690&#47;TeamandVenueAnalysis' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;IP&#47;IPLDescriptiveAnalysis_16550346868690&#47;TeamandVenueAnalysis&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1655122291257');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1277px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
    components.html(html_temp, width=1300, height=550, scrolling=True)
    

def playerarch():
    df_ball = pd.read_csv('IPL Ball-by-Ball 2008-2020.csv')
    df_match = pd.read_csv('IPL Matches 2008-2020.csv')
    batgroup = df_ball.groupby(['batsman'])

    # Create a batting dataframe with a summary statistics for each batsman
    batdf = pd.DataFrame(batgroup['ball'].count()).rename(columns={'ball':'balls_faced'})
    batdf['innings'] = batgroup['id'].nunique()
    batdf['runs'] = batgroup['batsman_runs'].sum()
    batdf['4s'] = df_ball[df_ball['batsman_runs'] == 4].groupby('batsman')['batsman_runs'].count()
    batdf['4s'].fillna(0,inplace=True)
    batdf['6s'] = df_ball[df_ball['batsman_runs'] == 6].groupby('batsman')['batsman_runs'].count()
    batdf['6s'].fillna(0,inplace=True)

    batdf['bat_average'] = round(batdf['runs']/batdf['innings'],2)

    batdf['bat_strike'] = round(batdf['runs']/batdf['balls_faced']*100,2)
        
    bowlgroup = df_ball.groupby(['bowler'])

    # Create a bowling dataframe (bowldf) with a summary statistics for each batsman
    bowldf = pd.DataFrame(bowlgroup['ball'].count()).rename(columns={'ball':'balls_bowled'})

    # Get no. of wickets taken by each bowler
    bwl_wkts = df_ball[df_ball['dismissal_kind'].isin(['caught','bowled', 'lbw','stumped', 'caught and bowled', 'hit wicket'])]
    bowldf['wickets'] = bwl_wkts.groupby(['bowler'])['ball'].count()
    bowldf['wickets'].fillna(0,inplace=True)

    # Calculate the total no. of overs bowled
    overs = pd.DataFrame(df_ball.groupby(['bowler','id'])['over'].nunique())
    bowldf['overs'] = overs.groupby(['bowler'])['over'].sum()    

    # Calculate the runs conceded
    bowldf['runs_conceded'] = df_ball.groupby('bowler')['batsman_runs'].sum()
    bowldf['runs_conceded'] = bowldf['runs_conceded'].fillna(0)
    # Add the runs conceded through wide and noball
    bowldf['runs_conceded'] = bowldf['runs_conceded'].add(df_ball[df_ball['extras_type'].isin(['wides','noballs'])].groupby('bowler')['extra_runs'].sum(),fill_value=0)
        
    # Add each player to the final all players list
    def update_player_with_match(player_name, id):
        if player_name in all_players_dict:
            all_players_dict[player_name].add(id)
        else:
            all_players_dict[player_name] = {id}

    def update_player_list(x):
        update_player_with_match(x['batsman'],x['id'])
        update_player_with_match(x['non_striker'],x['id'])
        update_player_with_match(x['bowler'],x['id']) 

        # Create a dataframe with all players list
    all_players_dict = {}
    out_temp = df_ball.apply(lambda x: update_player_list(x),axis=1)
    all_df = pd.DataFrame({'Players':list(all_players_dict.keys())})
    all_df['matches'] = all_df['Players'].apply(lambda x: len(all_players_dict[x]))
    all_df=all_df.set_index('Players')

    # Note - roughly apprx to overs.  Should be runs_conceded/overs.balls
    bowldf['bowl_econ'] = round(bowldf['runs_conceded']/bowldf['overs'],2)
        
    players = pd.merge(all_df,batdf, left_index=True, right_index=True,how='outer')
    players = pd.merge(players,bowldf, left_index=True, right_index=True,how='outer')

    players = pd.merge(players,df_match['player_of_match'].value_counts(), left_index=True, right_index=True,how='left')
    players['player_of_match']  = players[['player_of_match']].fillna(0)
        
    df_ball['dismissal_kind'].unique()
    # Total catches = Number of caught & bowled + number of catches as fielder
    # Capture caught & bowled instances
    catches_cb = df_ball[(df_ball['dismissal_kind'].isin(['caught and bowled']))].groupby('bowler') ['ball'].count().rename('bowler_catches')
    # Capture the catches
    catches_c = df_ball[(df_ball['dismissal_kind'].isin(['caught']))].groupby('fielder')['ball'].count().rename('fielder_catches')
        # Combine the caught & bowled and fielding catches to get the total catches.
    catches_df = pd.merge(catches_cb,catches_c, left_index=True, right_index=True,how='outer')
    catches_df.fillna(0,inplace=True)
    catches_df['catches'] = catches_df['bowler_catches']+catches_df['fielder_catches']
    catches_df.drop(['bowler_catches','fielder_catches'],axis=1,inplace=True)
    # Merge total catches to players data
    players = pd.merge(players,catches_df, left_index=True, right_index=True,how='outer')
    players.fillna(0,inplace=True)
        
    # Backup before data filtering
    all_players = players.copy()
    ALL_COLUMN_NAMES = list(players.columns)
        
    # Cut off on no. of matches
    # Current Analysis: Players who have played atleast 10 matches
    CUTOFF_MATCHES = 10
    players=players[players['matches']>=CUTOFF_MATCHES]
        
    # Standard Scaler (mean = 0 and standard deviation = 1)
    scaler = StandardScaler()

    # fit_transform
    players_scaled = pd.DataFrame(scaler.fit_transform(players),columns=players.columns)
        
    players.insert(0, 'name', players.index)
        
    NUM_CLUSTERS = 5
    # Initialize few colour codes for each cluster
    CLUSTER_COLORS = ['#F28A30','#0ABDA0','#008AC5','#D6618F','#F3CD05','#A882C1','#BDA589','#888C46',
                          '#36688D','#00743F','#0444BF','#A7414A','#1D65A6','red','green','blue','orange','pink','yellow']
        
    def print_with_column_color_style(input_df):
        col_bgcolor_dict = dict(zip(input_df.columns,['background-color:'+i for i in CLUSTER_COLORS]))
        def mycolor(x):
            return pd.DataFrame(col_bgcolor_dict,index=x.index,columns=x.columns)

        return input_df.style.apply(mycolor,axis=None)
    def  clustering_data(cluster_col):
        grouper = players.sort_values(["innings"], ascending = False)[['name',cluster_col]].groupby([cluster_col])
        cluster_df = pd.concat([pd.Series(v['name'].tolist(), name=k) for k, v in grouper], axis=1)
        cluster_df.fillna('',inplace=True)
        print("\n-----------------------\nPLAYERS IN EACH CLUSTER\n-----------------------\n")
#         with st.container():
        st.dataframe(data = (print_with_column_color_style(cluster_df)), width = 1000, height = 1000)

    from sklearn.cluster import KMeans

        # Define function to perform the kmeans clustering on the given data
    def kmeans_clustering(num_clusters, max_iterations,input_df,output_df, output_col):
        kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations)
        kmeans.fit(input_df)
        output_df[output_col] = kmeans.labels_ 
    # New output column to create for the cluster label
    kmeans_label = 'cluster_kmeans'

    # K-means clustering
    kmeans_clustering(5,50,players_scaled[ALL_COLUMN_NAMES],players,kmeans_label)
    clustering_data(kmeans_label)

def nav():
        st.markdown(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
        unsafe_allow_html=True,
    )
        query_params = st.experimental_get_query_params()
        tabs = ["Home", "Batsman Analysis", "Bowler Analysis","Team and Venue Analysis","Similar Player Finder"]
        if "tab" in query_params:
            active_tab = query_params["tab"][0]
        else:
            active_tab = "Home"

        if active_tab not in tabs:
            st.experimental_set_query_params(tab="Home")
            active_tab = "Home"

        li_items = "".join(
            f"""
            <li class="nav-item">
                <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}">{t}</a>
            </li>
            """
            for t in tabs
        )
        tabs_html = f"""
            <ul class="nav nav-tabs">
            {li_items}
            </ul>
        """

        st.markdown(tabs_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        #     #sidebar for menu
        #     with st.sidebar:
        #         st.markdown("**Performa** by RCB")
        #         menu = st.radio("Menu Options",('Home','Batsman Analysis','Bowler Analysis','Team Analysis')) 


        #calling all defined functions based on user input of menu
        if(active_tab == "Home"):
            title()
        elif(active_tab == "Batsman Analysis"):
            batsman()
        elif(active_tab == "Bowler Analysis"):
            bowler()
        elif(active_tab == "Team and Venue Analysis"):
            team()
        elif(active_tab == "Similar Player Finder"):
            playerarch()
        
    
#main function
if __name__ == "__main__":
    
    st.markdown("<h1 style = 'color:Black; font-family: Times, Serif; background-color:White; text-align:center; opacity:0.6'>  Performa by RCB </h1>",unsafe_allow_html=True)
    main_bg = "./pexels-jonathan-petersson-399187.jpg"
    main_bg_ext = "jpg"
    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
        opacity: 0.9
    }}
    </style>
    """,
    unsafe_allow_html=True
)
    nav()
        




