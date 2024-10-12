from datasets import Dataset, DatasetDict, Audio, Features, Value
import os
import librosa

# def create_dataset(audio_files, transcriptions):
#     # Get the audio durations
#     # durations = [librosa.get_duration(path=file) for file in audio_files]

#     # Create a dictionary with the data
#     data = {
#         "audio": audio_files,
#         "transcription": transcriptions,
#         # "duration": durations
#     }

#     # Define the features of the dataset
#     features = Features({
#         "audio": Audio(sampling_rate=16000),
#         "transcription": Value("string"),
#         # "duration": Value("float")
#     })

#     # Create the dataset
#     dataset = Dataset.from_dict(data, features=features)
#     return dataset
from datasets import Dataset, Features, Audio, Value

def create_dataset(audio_files, transcriptions):
    features = Features({
        "audio": Audio(sampling_rate=16000),
        "labels": Value("string")
    })

    dataset = Dataset.from_dict({
        "audio": audio_files,
        "labels": transcriptions
    }, features=features)

    return dataset
# Lists to store audio files and transcriptions
audio_files = []
transcriptions = []

# Read the label file
def load_txt(paths=['/media/sanslab/Data/DuyLong/whis/label.txt']):
    for path in paths:
        with open(path, 'r+') as file:
            for line in file:
                parts = line.strip().split('|')
                if len(parts)==2 or len(parts)==3:
                    audio_files.append(parts[0])
                    transcriptions.append(parts[1])
                    
    # Create the dataset
    dataset = create_dataset(audio_files, transcriptions)
    return dataset

# Print some information about the dataset

# print((load_txt()[0]['audio']['array']))