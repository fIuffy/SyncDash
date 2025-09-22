import librosa

def analyze_song(song_path):
    try:
        # Load the audio file with librosa
        y, sr = librosa.load(song_path, sr=None)

        # Get tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Ensure the beat_times array is not empty
        if tempo and len(beat_times) > 0:
            return tempo, beat_times
        else:
            print("No beats detected.")
            return None, None
    except Exception as e:
        print(f"Error analyzing song: {e}")
        return None, None

# Test
tempo, beat_times = analyze_song('../data/raw_songs/Griztronics-GRiZ_Subtronics.mp3')
if tempo is not None and beat_times is not None:
    print(f"Tempo: {tempo} BPM")
    print(f"Beat Times: {beat_times}")
else:
    print("Failed to analyze the song.")