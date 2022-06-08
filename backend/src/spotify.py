import spotipy

SPOTIFY_OAUTH_TOKEN = "BQCyRal1k_UJErTLkZyagOavTh-o3xb5yykcvh0ahP1aNc0XJ3tePxPxQCJFrmeqKgGEAcxpYaJQaH-z4-agxO8j1LTNuvRCgn4_StUvR_wMRhc53I-aFjIWIEbSDZn75mlWR7UPqMSwDYQ7c9R-WwdDNcUpDDn4PVEq"

spotify = spotipy.Spotify(auth=SPOTIFY_OAUTH_TOKEN)


def get_song_spotify_data(s):
    artist = s['metadata_songs_artist_name'].decode("utf-8")
    song = s['metadata_songs_title'].decode("utf-8")

    search_str = "artist:{} track:{} year:1920-2010".format(artist, song)

    try:
        result = spotify.search(search_str, limit=1)
    except:
        return 0,0
    

    if (len(result['tracks']['items']) < 1):
        return 0, 0
        
    track = result['tracks']['items'][0]
    
    year = int(track['album']
            ['release_date'].split('-')[0])
    popularity = track['popularity']

    return year, popularity
