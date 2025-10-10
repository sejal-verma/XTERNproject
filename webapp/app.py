import streamlit as st
import pandas as pd
import geopandas as gpd
import json
import numpy as np
import os
import plotly.express as px
import re
from collections import defaultdict
import datetime as dt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import praw

st.set_page_config(layout="wide")

#################
# load geo data
#################
os.makedirs("../data", exist_ok=True)

gdf_states_provinces = gpd.read_file("assets/ne_110m_admin_1_states_provinces.shp")
gdf_states_provinces = gdf_states_provinces.to_crs("EPSG:4326")

miso_names = [
    'Illinois', 'Indiana', 'Iowa', 'Kentucky', 'Louisiana',
    'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
    'North Dakota', 'South Dakota', 'Wisconsin'
]
gdf_miso = gdf_states_provinces[gdf_states_provinces['name'].isin(miso_names)]
gdf_miso['id'] = gdf_miso['postal']
gdf_miso['NAME'] = gdf_miso['name']

gdf_mb = gdf_states_provinces[gdf_states_provinces['name'] == 'Manitoba'].copy()
gdf_mb['id'] = 'MB'
gdf_mb['NAME'] = 'Manitoba'

gdf_all = pd.concat([gdf_miso, gdf_mb], ignore_index=True)
geojson_all = json.loads(gdf_all.to_json())

for feature in geojson_all['features']:
    feature['id'] = feature['properties']['id']

##################
# EIA MISO demand
##################
df_eia = pd.read_csv("../data/eia/eia_miso_demand_20251009.csv", parse_dates=["timestamp"])
df_eia['date'] = df_eia['timestamp'].dt.date
df_demand_summary = df_eia.groupby('respondent')['demand_MWh'].mean().reset_index()
df_demand_summary.rename(columns={'demand_MWh': 'Load'}, inplace=True)

state_to_fips = {
    'IL': 'IL', 'IN': 'IN', 'IA': 'IA', 'KY': 'KY', 'LA': 'LA',
    'MI': 'MI', 'MN': 'MN', 'MS': 'MS', 'MO': 'MO', 'ND': 'ND',
    'SD': 'SD', 'WI': 'WI', 'MB': 'MB'
}

df_demand_summary['county_fips'] = df_demand_summary['respondent'].map(state_to_fips)

if 'MB' not in df_demand_summary['county_fips'].values:
    df_demand_summary = pd.concat([
        df_demand_summary,
        pd.DataFrame({'respondent': ['MB'], 'Load': [0.0], 'county_fips': ['MB']})
    ], ignore_index=True)

############
# reddit
############
client_id = 'vRIO1v0I35RNbW3KAaR8Ow'
client_secret = 'yp-pAuofhdRx_mWJEyhljwpcZaACcg'
user_agent = 'yp-pAuofhdRx_mWJEyhljwpcZaACcg'

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

miso_subreddits = {
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KY': 'Kentucky', 'LA': 'Louisiana',
    'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'ND': 'NorthDakota', 'SD': 'SouthDakota', 'WI': 'Wisconsin', 'MB': 'Manitoba'
}

# energy categories
category_patterns = {
    "Resource Transition": r"transition|retire|retirement|decommission|coal plant|closure|shutdown",
    "Load Growth": r"load growth|data center|load increase|computing demand|industrial expansion",
    "Renewable Expansion": r"solar|wind|renewable|clean energy|geothermal|hydrogen",
    "Grid Modernization": r"transmission|distribution|grid upgrade|infrastructure|microgrid|resilience",
    "Storage & Reliability": r"battery|storage|capacity|reliability|backup power|peak load",
    "Market & Regulation": r"market|tariff|rate|pricing|policy|regulation|legislation|subsidy|incentive",
}

####################
# Reddit mentions and sentiment
####################
energy_keywords = list(set(
    [w for p in category_patterns.values() for w in re.findall(r"[a-zA-Z]+", p)]
))
energy_pattern = re.compile("|".join(energy_keywords), re.IGNORECASE)

reddit_counts = defaultdict(int)
sid = SentimentIntensityAnalyzer()

# mentions per state
for state_code, subreddit_name in miso_subreddits.items():
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.hot(limit=150):
            if energy_pattern.search(submission.title):
                reddit_counts[state_code] += 1
    except Exception as e:
        print(f"Error fetching r/{subreddit_name}: {e}")

df_reddit_metric = pd.DataFrame(list(reddit_counts.items()), columns=['state', 'reddit_mentions'])
df_reddit_metric['county_fips'] = df_reddit_metric['state'].map(state_to_fips)

# sentiment per category
reddit_sentiments = defaultdict(lambda: defaultdict(list))
for state_code, subreddit_name in miso_subreddits.items():
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.hot(limit=150):
            for cat, pattern in category_patterns.items():
                if re.search(pattern, submission.title, re.IGNORECASE):
                    reddit_sentiments[state_code][cat].append(
                        sid.polarity_scores(submission.title)['compound']
                    )
    except Exception as e:
        print(f"Error fetching sentiment r/{subreddit_name}: {e}")

reddit_sent_avg = []
for state, cat_dict in reddit_sentiments.items():
    for cat, scores in cat_dict.items():
        avg_score = np.mean(scores) if scores else 0
        reddit_sent_avg.append({
            'state': state,
            'category': cat,
            'reddit_sentiment': avg_score,
            'county_fips': state_to_fips[state]
        })
df_reddit_sentiment = pd.DataFrame(reddit_sent_avg)

########################
# openstates and legiscan
########################
def categorize_title(title):
    for cat, pattern in category_patterns.items():
        if re.search(pattern, str(title), re.IGNORECASE):
            return cat
    return "Other"

# load sources
df_open = pd.read_csv("../data/bills/openstates_energy_20251008.csv", parse_dates=["latest_action_date"])
df_open['state'] = df_open['jurisdiction_id'].str.extract(r'state:([a-z]{2})', expand=False).str.upper()

df_legiscan = pd.read_csv("../data/bills/legiscan_energy_20251008.csv", parse_dates=["date"])
df_legiscan['state'] = df_legiscan['state'].astype(str).str.upper()

# categorization and sentiment
for df in [df_open, df_legiscan]:
    df['category'] = df['title'].apply(categorize_title)
    df['sentiment'] = df['title'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

def aggregate_sentiment(df, source_name):
    df_filtered = df[df['category'] != "Other"]
    grouped = df_filtered.groupby(['state', 'category'], as_index=False)['sentiment'].mean()
    grouped.rename(columns={'sentiment': f'sentiment_{source_name}'}, inplace=True)
    grouped['county_fips'] = grouped['state'].map(state_to_fips)
    return grouped

df_openstates_sent = aggregate_sentiment(df_open, 'openstates')
df_legiscan_sent = aggregate_sentiment(df_legiscan, 'legiscan')

# merge sources
df_legislation_sentiment = pd.merge(
    df_openstates_sent,
    df_legiscan_sent,
    on=['state', 'category', 'county_fips'],
    how='outer'
)
df_legislation_sentiment['sentiment_legislation'] = df_legislation_sentiment[
    ['sentiment_openstates', 'sentiment_legiscan']
].mean(axis=1)

#########################
# bill counts per state
#########################
df_state_counts = pd.concat([
    df_open.groupby('state').size().reset_index(name='Bills_StateLevel'),
    df_legiscan.groupby('state').size().reset_index(name='Bills_StateLevel')
], ignore_index=True)
df_state_counts = df_state_counts.groupby('state', as_index=False)['Bills_StateLevel'].sum()
df_state_counts['county_fips'] = df_state_counts['state'].map(state_to_fips)

################
# merge metrics
################
df_metrics = pd.DataFrame({'county_fips': list(state_to_fips.values())})
df_metrics = df_metrics.merge(df_demand_summary[['county_fips', 'Load']], on='county_fips', how='left')
df_metrics = df_metrics.merge(df_reddit_metric[['county_fips', 'reddit_mentions']], on='county_fips', how='left')
df_metrics = df_metrics.merge(df_state_counts[['county_fips', 'Bills_StateLevel']], on='county_fips', how='left')
df_metrics.fillna(0, inplace=True)

fips_to_name = {row['id']: row['NAME'] for _, row in gdf_all.iterrows()}
df_metrics['NAME'] = df_metrics['county_fips'].map(fips_to_name)

#############
# streamlit
#############
st.title("MISO Energy Landscape Dashboard")

category_choice = st.selectbox("Select category:", list(category_patterns.keys()))

metric_choice = st.selectbox(
    "Select metric to display:",
    ["Load", "reddit_mentions", "Bills_StateLevel", "Reddit Sentiment", "Legislation Sentiment"]
)

if metric_choice == "Reddit Sentiment":
    df_plot = df_metrics[['county_fips', 'NAME']].merge(
        df_reddit_sentiment[df_reddit_sentiment['category'] == category_choice][['county_fips', 'reddit_sentiment']],
        on='county_fips', how='left'
    )
    color_col = 'reddit_sentiment'

elif metric_choice == "Legislation Sentiment":
    df_plot = df_metrics[['county_fips', 'NAME']].merge(
        df_legislation_sentiment[df_legislation_sentiment['category'] == category_choice][['county_fips', 'sentiment_legislation']],
        on='county_fips', how='left'
    )
    color_col = 'sentiment_legislation'

else:
    color_col = metric_choice
    df_plot = df_metrics.copy()

df_plot[color_col] = df_plot[color_col].fillna(0)

#############
# plotly
#############
fig = px.choropleth_map(
    df_plot,
    geojson=geojson_all,
    locations='county_fips',
    color=color_col,
    hover_name='NAME',
    color_continuous_scale="Viridis",
    map_style="carto-positron",
    center={"lat": 50, "lon": -95},
    zoom=3,
    opacity=0.7
)

fig.add_annotation(
    text=f"Category: {category_choice}",
    xref="paper", yref="paper",
    x=0.02, y=0.02, showarrow=False,
    bgcolor="rgba(255,255,255,0.7)", bordercolor="gray", borderwidth=1
)

fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

#############
# Info table
#############
st.subheader("Legislation Sentiment by State & Category")
st.dataframe(df_legislation_sentiment.sort_values(['state', 'category']))
