import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, BatchNormalization, Activation, Softmax
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
from keras.optimizer_v2.adadelta import Adadelta as Adadelta
from sklearn import metrics
import matplotlib.pyplot as plt

import preprocessing as pp
import read_h5 as read
from spotify import get_song_spotify_data

initial_df = read.h5_to_df('/mnt/snap/data')

df = initial_df.copy()

df.drop(df[df['musicbrainz_songs_year'] < 1].index, inplace=True)


def apply_periodicity(df, period):
    sinus_name = "year_sin_{}".format(period)
    df[sinus_name] = df.apply(lambda row: math.sin(
        period * 2 * math.pi / row['musicbrainz_songs_year']), axis=1)

    return df


def preprocess_columns(df):
    df = df.loc[:, ['analysis_bars_confidence', 'analysis_bars_start',
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
                    'metadata_songs_song_hotttnesss',
                    'musicbrainz_songs_year'
                    ]]

    df['target'] = df['metadata_songs_song_hotttnesss'].apply(
        lambda x: 1 if x > 0.5 else 0)
    df.drop(['metadata_songs_song_hotttnesss'], inplace=True, axis=1)

    # mode_song_year = df["musicbrainz_songs_year"].median()
    # df["musicbrainz_songs_year"].replace(0, mode_song_year, inplace=True)

    df.drop(df[df['musicbrainz_songs_year'] < 1].index, inplace=True)

    # Добавить периодичность годов от каждого 2 до каждого 10
    for i in range(2, 100):
        df = apply_periodicity(df, i)

    return df


df = preprocess_columns(initial_df.copy())

# Shuffle a few times
for i in range(5):
    df = df.iloc[np.random.permutation(len(df))]
df = df.fillna(0)

# Transform into NumPy matrix, normalized by column
X, y, y_map = pp.vectorize(df, 'target')


# Globals
lr = 0.01
epochs = 50
batch_size = 100
OPTIMIZER_NAME = 'adadelta'

# Calculate class weights to improve accuracy
class_weights = dict(
    enumerate(compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)))
swm = np.array([class_weights[i] for i in y])

classes_count = len(class_weights)
# Convert target to categorical
y = to_categorical(y=y, num_classes=classes_count)

# Размер входа равен размерности данных
in_size = X.shape[1]

# Размеры скрытых слоев
hidden_1_size = 400
hidden_2_size = 150
hidden_3_size = 50

out_size = classes_count


# Разделение на train/test/validation
X_train, X_test, X_valid = np.split(
    X, [int(.6 * len(X)), int(.8 * len(X))])

y_train, y_test, y_valid = np.split(
    y, [int(.6 * len(y)), int(.8 * len(y))])


def add_hidden_layer(s, b=False, a=0.3, d=False):
    if d:
        model.add(Dense(s, kernel_initializer='normal',
                        kernel_constraint=maxnorm(3)))
    else:
        model.add(Dense(s))

    if b:
        model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=a))


model = Sequential()

# Входной слой
model.add(Dense(in_size, input_shape=(in_size,), activation='relu',
                kernel_initializer='normal', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))

# Скрытые слои
add_hidden_layer(in_size, False, 0.3, True)
add_hidden_layer(hidden_1_size, True, 0.3)
add_hidden_layer(hidden_2_size, True, 0.3, True)
add_hidden_layer(hidden_3_size, False, 0.3, True)

# Выходной слой
model.add(Dense(out_size))
model.add(Softmax(axis=-1))


optimizer = Adadelta(learning_rate=0.8, rho=0.95, epsilon=None, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', 'msle'],
              sample_weight_mode=swm
              )

model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
          epochs=epochs, batch_size=batch_size, verbose=1,
          shuffle=True)

y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test, verbose=1)

print(score)

print('Saving model...')
path = '/var/v1'
# Save model structure json
model_json = model.to_json()
with open(path + '/model.json', 'w') as file:
    file.write(model_json)
# Save weights as h5
model.save_weights(path + '/weights.h5')
# Save sample weight mode
np.savetxt(path + '/sample_weights.csv', swm, delimiter=',')
# Save hyperparams
with open(path + '/hyperparams.csv', 'w') as file:
    file.write(','.join([str(lr), OPTIMIZER_NAME]))
print('Model saved to disk')

fpr, tpr, _ = metrics.roc_curve(
    np.argmax(y_test, axis=-1), np.argmax(y_pred, axis=-1))

print(fpr)
# create ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
