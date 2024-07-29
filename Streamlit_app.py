import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import euclidean_distances
import re
import joblib

import streamlit as st
from PIL import Image
import requests
from io import BytesIO

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API credentials
client_id = '[CLIENT ID]'
client_secret = '[CLIENT SECRET]'

# Initialize Spotipy with user credentials
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

def predict_songs(song, artist_main):
    data_df = pd.read_csv('SpotifyDataset/data.csv')

    data = data_df.drop_duplicates(subset=['artists', 'name']) # dropping duplicates
    data = data.loc[data['name'].str.contains("[a-zA-Z]+"), :]
    data.loc[:, 'artists_main'] = data['artists'].apply(lambda i: re.search(r"([\w|\s]+)", i).group(0))
    data.loc[:, 'artists_num'] = data['artists'].str.count("(['\w\s\d]+)+")
    data = data.loc[data['year'] >= 1950, :]
    X = data.drop(columns=['id', 'key', 'name', 'year', 'release_date', 'artists'])

    joblib_file = "Models/song_recommendation_pipeline.pkl"

    pipe = joblib.load(joblib_file)

    X_norm = pipe['column_transformer'].fit_transform(X)
    X_pca = pipe['pca'].fit_transform(X_norm)

    data[['pca_1', 'pca_2']] = X_pca
    data['Cluster'] = pipe.fit_predict(X)
    # song = 'Break It Off'
    # artist_main = 'Rihanna'
    n_cluster = data.loc[data['name'] == song, 'Cluster'].values[0]
    df_music = data.loc[(data['name'] == song) & (data['artists_main'] == artist_main), ['pca_1', 'pca_2']]
    group = data.loc[data['Cluster'] == n_cluster, :]
    group.loc[:, 'distances'] = euclidean_distances(group[['pca_1', 'pca_2']], df_music)
    op_df = group[['artists', 'name', 'distances', 'pca_1', 'pca_2']].sort_values('distances', ascending=True).head(6)
    op_df = op_df[1:]
    return op_df


def fetch_song_info(song_name, artist_name):
    # Search for the song by name and artist
    results = sp.search(q=f'track:{song_name} artist:{artist_name}', type='track', limit=1)
    
    if results['tracks']['items']:
        with st.spinner('Wait for it...'):
            track = results['tracks']['items'][0]
            song_info = {
                'song_name': track['name'],
                'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                'album_name': track['album']['name'],
                'album_image': track['album']['images'][0]['url'] if track['album']['images'] else None,
                'song_link': track['external_urls']['spotify']
            }
            return song_info
    else:
        return None
    
if __name__ =='__main__':
    st.set_page_config(page_title="The AI DJ", page_icon="🎧", layout="wide")    
    st.title('The AI DJ 🤖')

    # Input fields for song name and artist name
    song_name = st.text_input('Song Name')
    artist_name = st.text_input('Artist Name')

    # Predict button
    if st.button('Surprise me!'):
        st.write(f"Top 5 Recommended Songs that matches {song_name} by {artist_name}:")
        op_df = predict_songs(song_name, artist_name)
        predicted_song_info = {}
        for artist, song in zip(list(op_df['artists']), list(op_df['name'])):
            song_info = fetch_song_info(song, artist)

            if song_info:
                predicted_song_info[f"{song}_{artist}"] = song_info
            else:
                predicted_song_info[f"{song}_{artist}"] = "Info Not Found"
        print((predicted_song_info))
        cols = st.columns(len(predicted_song_info))
        i = 0
        song_link = "https://open.spotify.com/"
        for values in predicted_song_info.values():
            # print(values['song_name'])
            if values['song_name'] and values['artist_name']:
                # st.write(values['song_name'], values['artist_name'])
                # Display the images
                if values['album_image']:
                    img_url = values['album_image']
                    song_link = values['song_link']
                response = requests.get(img_url)
                img = Image.open(BytesIO(response.content))
                cols[i].image(img, use_column_width=True)
                cols[i].write(f"{values['song_name']} by - {values['artist_name']}")
                cols[i].write(f"[Listen Now!]({song_link})")
            else:
                st.write("No recommendations found.")
            i+=1 
    else:
        st.write("Please enter song and artist name.")
