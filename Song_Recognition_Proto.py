import mysql.connector
from mysql.connector import Error
import pyaudio
import numpy as np
import librosa
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from dtw import *
import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# Configure logging
logging.basicConfig(level=logging.INFO, filename='song_recognition.log', 
                    filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Spotify API setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='66d88cbbcd8542e1b5cde614af4d6676',
                                                           client_secret='3862e16fd4eb4f39807d795f7918289a'))

# MySQL connection function
def create_connection():
    """ Creating a database connection """
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',  
            database='spotify_recommendations'
        )
        if connection.is_connected():
            print("Connection to MySQL DB successful")
            logging.info("Connection to MySQL DB successful")
    except Error as e:
        logging.error(f"The error '{e}' occurred")
        print(f"The error '{e}' occurred")
    return connection


# Extract features from audio
def extract_features(audio, sample_rate=None, n_mfcc=13):
    """ Extract MFCC, chroma, and spectral contrast features from audio data """
    try:
        if sample_rate is None:
            y, sr = librosa.load(audio, duration=20)  # Load a 20-second snippet from file
        else:
            y, sr = audio, sample_rate  # Use the provided raw audio data and sample rate
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)

        # Chroma feature
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)

        # Concatenate all features into one feature vector 
        return np.hstack((mfcc_mean, spectral_contrast, chroma))
    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        print(f"Error processing audio: {e}")
        return np.zeros(n_mfcc + 7 + 12)  # Return a default value in case of error (total feature length)


# Recording and processing audio (for capturing user audio sample)
def record_audio(duration=20, sample_rate=44100):
    """ Record audio from microphone """
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")
    logging.info("Recording audio...")

    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Finished recording.")
    logging.info("Finished recording audio.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.hstack(frames)
    return audio_data, sample_rate


# Function to store song metadata in MySQL database
def store_song(connection, song):
    """ Store song metadata in the database, including Spotify track ID """
    if not connection:
        logging.error("No valid database connection.")
        print("No valid database connection.")
        return

    if not song or not isinstance(song, dict):
        logging.error("Invalid song data.")
        print("Invalid song data.")
        return

    # Ensure 'spotify_track_id' key exists in song
    if 'spotify_track_id' not in song:
        song['spotify_track_id'] = None  # Default value if not available

    cursor = connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO songs (song_id, title, artist, genre, popularity, path, spotify_track_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE title=%s, artist=%s, genre=%s, popularity=%s, path=%s, spotify_track_id=%s
        """, (song['song_id'], song['title'], song['artist'], song['genre'], song['popularity'], song['path'], song['spotify_track_id'],
              song['title'], song['artist'], song['genre'], song['popularity'], song['path'], song['spotify_track_id']))
        connection.commit()
        logging.info(f"Song '{song['title']}' stored successfully.")
        print(f"Song '{song['title']}' stored successfully.")
    except Error as e:
        logging.error(f"Error storing song: {e}")
        print(f"Error storing song: {e}")
        connection.rollback()
    finally:
        cursor.close()


# Function to search for a track on Spotify
def search_track(query):
    """Search for a track on Spotify using a query"""
    try:
        result = sp.search(q=query, type='track', limit=1)
        if result['tracks']['items']:
            track = result['tracks']['items'][0]
            return {
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'release_date': track['album']['release_date'],
                'duration_ms': track['duration_ms'],
                'popularity': track['popularity'],
                'spotify_url': track['external_urls']['spotify']
            }
        else:
            print("No track found for the given query.")
            return None
    except Exception as e:
        print(f"Error searching track: {e}")
        return None


# Update songs with Spotify track IDs
def update_songs_with_spotify_ids(connection, songs):
    """ Update existing songs in the database with Spotify track IDs """
    for song in songs:
        query = f"{song['title']} {song['artist']}"
        track_info = search_track(query)
        if track_info:
            song['spotify_track_id'] = track_info['id']
            store_song(connection, song)  # Store or update song with track ID
        else:
            print(f"Track not found for {query}")


# Get recommendation based on user's favorite genre
def get_recommendation(connection, user_id):
    """ Get song recommendation based on user's favorite genre """
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute("SELECT favorite_genre FROM users WHERE user_id = %s", (user_id,))
        user = cursor.fetchone()

        if user:
            favorite_genre = user['favorite_genre']
            cursor.execute("SELECT * FROM songs WHERE genre = %s", (favorite_genre,))
            songs = cursor.fetchall()

            if songs:
                df = pd.DataFrame(songs)
                df = df.sort_values(by='popularity', ascending=False)
                recommended_song = df.iloc[0]

                logging.info(f"Recommended song: {recommended_song['title']} by {recommended_song['artist']}")
                print(f"Recommended song: {recommended_song['title']} by {recommended_song['artist']}")
            else:
                logging.warning("No songs found for the favorite genre.")
                print("No songs found for the favorite genre.")
        else:
            logging.error(f"User with ID {user_id} not found.")
            print(f"User with ID {user_id} not found.")
    except Error as e:
        logging.error(f"Error retrieving recommendations: {e}")
        print(f"Error retrieving recommendations: {e}")
    finally:
        cursor.close()


# Train model with additional features and more neighbors
def train_model(songs):
    """ Train a k-NN model using MFCC, spectral contrast, and chroma features """
    features = []
    labels = []
    song_ids = []

    for song in songs:
        mfcc = extract_features(song['path'])
        if not np.array_equal(mfcc, np.zeros(32)):  # Ensure valid features
            features.append(mfcc)
            labels.append(song['song_id'])
            song_ids.append(song['song_id'])

    if features:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)  # Normalize features

        knn = NearestNeighbors(n_neighbors=8)  # Increase neighbors to 8
        knn.fit(features, song_ids)
        logging.info("k-NN model trained successfully.")
        return knn, song_ids, scaler
    else:
        logging.error("No valid features to train the model.")
        print("No valid features to train the model.")
        return None, [], None


# Detect song with DTW and more accurate matching
def detect_song(knn_model, song_ids, scaler, connection):
    """ Detect a song based on user-recorded audio using DTW and enhanced features """
    audio_data, sample_rate = record_audio()
    features = extract_features(audio_data, sample_rate).reshape(1, -1)

    if knn_model:
        features = scaler.transform(features)  # Normalize the new input features
        distances, indices = knn_model.kneighbors(features)

        # Check the top 2 nearest neighbors
        for i in range(2):
            song_id = song_ids[indices[0][i]]
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute("SELECT * FROM songs WHERE song_id = %s", (song_id,))
                songs = cursor.fetchall()  # Fetch all results to avoid the unread result error

                if songs:  # Ensure there is at least one result
                    song = songs[0]  # Since we expect only one result, take the first one
                    print(f"Detected possible match: {song['title']} by {song['artist']}")

                    # Perform Dynamic Time Warping (DTW) for more accurate matching
                    song_features = extract_features(song['path'])
                    dtw_obj = dtw(features.flatten(), song_features.flatten())
                    dtw_distance = dtw_obj.distance

                    if dtw_distance < 500:  # Set a more precise threshold for DTW distance
                        print(f"approx match detected: {song['title']} by {song['artist']}")
                        logging.info(f"approx match detected: {song['title']} by {song['artist']}")
                        return  # Stop if a match is found
                    
                    if dtw_distance < 50:  # Set a more precise threshold for DTW distance
                        print(f"Exact match detected: {song['title']} by {song['artist']}")
                        logging.info(f"Exact match detected: {song['title']} by {song['artist']}")
                        return  # Stop if a match is found
            except Error as e:
                logging.error(f"Error during song detection: {e}")
                print(f"Error during song detection: {e}")
            finally:
                cursor.close()  # Ensure the cursor is always closed, even if an exception occurs
        else:
            print("Song not found in database, you have found a new song...")
            logging.info("Song not found in database, detected new song.")
    else:
        logging.error("No k-NN model available for song detection.")
        print("No k-NN model available for song detection.")





# Function to display all songs in the database
def display_all_songs(connection):
    """ Display all songs from the database """
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM songs")
        songs = cursor.fetchall()
        if songs:
            for song in songs:
                print(f"Song ID: {song['song_id']}, Title: {song['title']}, Artist: {song['artist']}")
        else:
            print("No songs available in the database.")
    except Error as e:
        logging.error(f"Error fetching songs: {e}")
        print(f"Error fetching songs: {e}")
    finally:
        cursor.close()

        


# usage with more songs and database connections
if __name__ == "__main__":
    connection = create_connection()

    # songs list with file paths
    songs = [
        {'song_id': 1, 'title': 'What Makes You Beautiful', 'artist': 'One Direction', 'genre': 'Pop', 'popularity': 85, 'path': 'C:\\Users\\nithi\\Music\\What_Makes_You_Beautiful.mp3'},
        {'song_id': 2, 'title': 'You And I', 'artist': 'One Direction', 'genre': 'pop', 'popularity':75, 'path': 'C:\\Users\\nithi\\Music\\You___I.mp3'},
        {'song_id': 3, 'title': 'The Story Never Ends', 'artist': 'Lauv', 'genre': 'melody', 'popularity': 90, 'path': 'C:\\Users\\nithi\\Music\\The Story Never Ends.mp3'},
        {'song_id': 4, 'title': 'Life Is A Highway', 'artist': 'Rascal Flatts', 'genre': 'rock', 'popularity': 94, 'path': 'C:\\Users\\nithi\\Music\\Life is a Highway.mp3'},
        {'song_id': 5, 'title': 'Heat Waves', 'artist': 'Glass Animals', 'genre': 'rock', 'popularity': 100, 'path': 'C:\\Users\\nithi\\Music\\Heat_Waves.mp3'},
        {'song_id': 6, 'title': 'Double Take', 'artist': 'Dhruv', 'genre': 'classical', 'popularity': 80, 'path': 'C:\\Users\\nithi\\Music\\double_take.mp3'},
        {'song_id': 7, 'title': 'Attaintion', 'artist': 'Charlie Puth', 'genre': 'Pop', 'popularity': 50, 'path': 'C:\\Users\\nithi\\Music\\Attention.mp3'},
        {'song_id': 8, 'title': 'Arcade', 'artist': 'Duncan Laurence', 'genre': 'rock', 'popularity': 79, 'path': 'C:\\Users\\nithi\\Music\\Arcade.mp3'},
        {'song_id': 9, 'title': 'Sunflower', 'artist': 'Post Malone', 'genre': 'pop', 'popularity': 100, 'path': 'C:\\Users\\nithi\\Music\\Sunflower - Spider-Man_ Into the Spider-Verse.mp3'},
        {'song_id': 10, 'title': 'Glimpse of Us', 'artist': 'Joji ', 'genre': 'R&B and soul', 'popularity':100, 'path': 'C:\\Users\\nithi\\Music\\Glimpse_of_Us.mp3'}
        # Add more songs as needed
    ]

    # Update songs with Spotify track IDs
    update_songs_with_spotify_ids(connection, songs)

    # Store songs in the database
    for song in songs:
        store_song(connection, song)

    # Train the k-NN model
    knn_model, song_ids, scaler = train_model(songs)

    # Detect a song
    detect_song(knn_model, song_ids, scaler, connection)


    # Display all songs
    #display_all_songs(connection)          #(use this if you need to display all the songs in the database)


    # Get recommendation
    get_recommendation(connection, user_id=1)

    # Close the connection when done
    if connection.is_connected():
        connection.close()








