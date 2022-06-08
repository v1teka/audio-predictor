
from datetime import datetime
import os
import pandas as pd
import numpy as np

columns = ['analysis_bars_confidence', 'analysis_bars_start',
           'analysis_beats_confidence', 'analysis_beats_start',
           'analysis_sections_confidence', 'analysis_sections_start',
           'analysis_segments_confidence', 'analysis_segments_loudness_max',
           'analysis_segments_loudness_max_time',
           'analysis_segments_loudness_start', 'analysis_segments_pitches',
           'analysis_segments_start', 'analysis_segments_timbre',
           'analysis_songs_analysis_sample_rate',
           'analysis_songs_danceability', 'analysis_songs_duration',
           'analysis_songs_end_of_fade_in', 'analysis_songs_energy',
           'analysis_songs_idx_bars_confidence', 'analysis_songs_idx_bars_start',
           'analysis_songs_idx_beats_confidence', 'analysis_songs_idx_beats_start',
           'analysis_songs_idx_sections_confidence',
           'analysis_songs_idx_sections_start',
           'analysis_songs_idx_segments_confidence',
           'analysis_songs_idx_segments_loudness_max',
           'analysis_songs_idx_segments_loudness_max_time',
           'analysis_songs_idx_segments_loudness_start',
           'analysis_songs_idx_segments_pitches',
           'analysis_songs_idx_segments_start',
           'analysis_songs_idx_segments_timbre',
           'analysis_songs_idx_tatums_confidence',
           'analysis_songs_idx_tatums_start',
           'analysis_songs_key',
           'analysis_songs_key_confidence', 'analysis_songs_loudness',
           'analysis_songs_mode', 'analysis_songs_mode_confidence',
           'analysis_songs_start_of_fade_out', 'analysis_songs_tempo',
           'analysis_songs_time_signature',
           'analysis_songs_time_signature_confidence',
           'analysis_tatums_confidence', 'analysis_tatums_start',
           'metadata_songs_artist_location',
           'metadata_songs_artist_longitude',
           'metadata_songs_artist_latitude',
           'musicbrainz_songs_year']


def get_features(file_data):
    return pd.read_csv('src/asd.csv')
    return {
        'analysis_bars_confidence': analysis_bars_confidence(file_data),
        'analysis_bars_start': analysis_bars_start(file_data),
        'analysis_beats_confidence': analysis_beats_confidence(file_data),
        'analysis_bars_confidence': analysis_bars_confidence(file_data),
        'musicbrainz_songs_year': musicbrainz_songs_year(),
        'metadata_songs_artist_latitude': metadata_songs_artist_latitude(file_data),
        'metadata_songs_artist_longitude': metadata_songs_artist_longitude(file_data),
        'metadata_songs_artist_location': metadata_songs_artist_location(file_data),
        'analysis_songs_danceability': analysis_songs_danceability(file_data),
        'analysis_beats_start': analysis_beats_start(file_data),
        'analysis_sections_confidence': analysis_sections_confidence(file_data),
        'analysis_sections_start': analysis_sections_start(file_data),
        'analysis_segments_confidence': analysis_segments_confidence(file_data),
        'analysis_segments_loudness_max': analysis_segments_loudness_max(file_data),
        'analysis_segments_loudness_max_time': analysis_segments_loudness_max_time(file_data),
        'analysis_segments_loudness_start': analysis_segments_loudness_start(file_data),
        'analysis_segments_pitches': analysis_segments_pitches(file_data),
        'analysis_segments_start': analysis_segments_start(file_data),
        'analysis_segments_timbre': analysis_segments_timbre(file_data),
        'analysis_songs_analysis_sample_rate': analysis_songs_analysis_sample_rate(file_data),
        'analysis_songs_duration': analysis_songs_duration(file_data),
        'analysis_songs_end_of_fade_in': analysis_songs_end_of_fade_in(file_data),
        'analysis_songs_energy': analysis_songs_energy(file_data),
        'analysis_songs_idx_bars_confidence': analysis_songs_idx_bars_confidence(file_data),
        'analysis_songs_idx_bars_start': analysis_songs_idx_bars_start(file_data),
        'analysis_songs_idx_beats_confidence': analysis_songs_idx_beats_confidence(file_data),
        'analysis_songs_idx_beats_start': analysis_songs_idx_beats_start(file_data),
        'analysis_songs_idx_sections_confidence': analysis_songs_idx_sections_confidence(file_data),
        'analysis_songs_idx_sections_start': analysis_songs_idx_sections_start(file_data),
        'analysis_songs_idx_segments_confidence': analysis_songs_idx_segments_confidence(file_data),
        'analysis_songs_idx_segments_loudness_max': analysis_songs_idx_segments_loudness_max(file_data),
        'analysis_songs_idx_segments_loudness_max_time': analysis_songs_idx_segments_loudness_max_time(file_data),
        'analysis_songs_idx_segments_loudness_start': analysis_songs_idx_segments_loudness_start(file_data),
        'analysis_songs_idx_segments_pitches': analysis_songs_idx_segments_pitches(file_data),
        'analysis_songs_idx_segments_start': analysis_songs_idx_segments_start(file_data),
        'analysis_songs_idx_segments_timbre': analysis_songs_idx_segments_timbre(file_data),
        'analysis_songs_idx_tatums_confidence': analysis_songs_idx_tatums_confidence(file_data),
        'analysis_songs_idx_tatums_start': analysis_songs_idx_tatums_start(file_data),
        'analysis_songs_key': analysis_songs_key(file_data),
        'analysis_songs_key_confidence': analysis_songs_key_confidence(file_data),
        'analysis_songs_loudness': analysis_songs_loudness(file_data),
        'analysis_songs_mode': analysis_songs_mode(file_data),
        'analysis_songs_mode_confidence': analysis_songs_mode_confidence(file_data),
        'analysis_songs_start_of_fade_out': analysis_songs_start_of_fade_out(file_data),
        'analysis_songs_tempo': analysis_songs_tempo(file_data),
        'analysis_songs_time_signature': analysis_songs_time_signature(file_data),
        'analysis_songs_time_signature_confidence': analysis_songs_time_signature_confidence(file_data),
        'analysis_tatums_confidence': analysis_tatums_confidence(file_data),
        'analysis_tatums_start': analysis_tatums_start(file_data)
    }


def analysis_bars_confidence(file_data):
    return np.array([0.716, 0.306, 0.015, 0.572, 0.529, 0.353, 0.206, 0.937, 0.124,
            0.612, 0.233, 0.217, 0.331, 0.755, 0.877, 0.417, 0.335, 0.525,
            0.011, 0.984, 0.964, 0.56, 0.227, 0.484, 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.])


def analysis_bars_start(file_data):
    return np.array([8.69527,  12.01758,  14.81363,  20.09003,  22.69399,  25.33891,
            28.16483,  34.69648,  37.57769,  40.45239,  43.36193,  46.32883,
            49.36182,  52.28227,  55.29653,  58.39561,  61.33526,  64.33449,
            67.39918,  70.39778,  73.34968,  76.49926,  79.48067,  82.50244,
            85.30177,  88.45055,  91.63519, 119.37849, 121.90584, 124.94568,
            128.01232, 130.93554, 133.9795, 137.07232, 140.09803, 143.06125,
            146.44263, 149.45155, 152.52088, 155.47345, 158.49279, 161.45351,
            164.39593, 167.37575, 170.44872, 173.42656, 176.19989, 179.1868,
            182.21737, 185.16192, 187.98025, 191.11476, 194.07118, 196.99324,
            200.0051, 203.73719, 206.86526, 209.95061, 213.20664, 216.69141,
            219.85238, 224.74045, 227.7615, 230.7771, 233.77406, 236.76481,
            239.50401, 242.43163, 245.42238])


def analysis_beats_confidence(file_data):
    return np.array([0.716, 0.306, 0.015, 0.572, 0.529, 0.353, 0.206, 0.937, 0.124,
            0.612, 0.233, 0.217, 0.331, 0.755, 0.877, 0.417, 0.335, 0.525,
            0.011, 0.984, 0.964, 0.56, 0.227, 0.484, 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.])


def analysis_beats_start(file_data):
    return 0


def analysis_sections_confidence(file_data):
    return 0


def analysis_sections_start(file_data):
    return 0


def analysis_segments_confidence(file_data):
    return 0


def analysis_segments_loudness_max(file_data):
    return 0


def analysis_segments_loudness_max_time(file_data):
    return 0


def analysis_segments_loudness_start(file_data):
    return 0


def analysis_segments_pitches(file_data):
    return 0


def analysis_segments_start(file_data):
    return 0


def analysis_segments_timbre(file_data):
    return 0


def analysis_songs_analysis_sample_rate(file_data):
    return 0


def analysis_songs_danceability(file_data):

    return 0.4


def analysis_songs_duration(file_data):
    return 0


def analysis_songs_end_of_fade_in(file_data):
    return 0


def analysis_songs_energy(file_data):
    return 0


def analysis_songs_idx_bars_confidence(file_data):
    return 0


def analysis_songs_idx_bars_start(file_data):
    return 0


def analysis_songs_idx_beats_confidence(file_data):
    return 0


def analysis_songs_idx_beats_start(file_data):
    return 0


def analysis_songs_idx_sections_confidence(file_data):
    return 0


def analysis_songs_idx_sections_start(file_data):
    return 0


def analysis_songs_idx_segments_confidence(file_data):
    return 0


def analysis_songs_idx_segments_loudness_max(file_data):
    return 0


def analysis_songs_idx_segments_loudness_max_time(file_data):
    return 0


def analysis_songs_idx_segments_loudness_start(file_data):
    return 0


def analysis_songs_idx_segments_pitches(file_data):
    return 0


def analysis_songs_idx_segments_start(file_data):
    return 0


def analysis_songs_idx_segments_timbre(file_data):
    return 0


def analysis_songs_idx_tatums_confidence(file_data):
    return 0


def analysis_songs_idx_tatums_start(file_data):
    return 0


def analysis_songs_key(file_data):
    return 0


def analysis_songs_key_confidence(file_data):
    return 0


def analysis_songs_loudness(file_data):
    return 0


def analysis_songs_mode(file_data):
    return 0


def analysis_songs_mode_confidence(file_data):
    return 0


def analysis_songs_start_of_fade_out(file_data):
    return 0


def analysis_songs_tempo(file_data):
    return 0


def analysis_songs_time_signature(file_data):
    return 0


def analysis_songs_time_signature_confidence(file_data):
    return 0


def analysis_tatums_confidence(file_data):
    return 0


def analysis_tatums_start(file_data):
    return 0


def metadata_songs_artist_location(file_data):
    return 'Moscow, Russia'  # Moscow values by default


def metadata_songs_artist_longitude(file_data):
    return 37.618423  # Moscow values by default


def metadata_songs_artist_latitude(file_data):
    return 55.751244  # Moscow values by default


def musicbrainz_songs_year():
    return datetime.now().year  # Current time by default
