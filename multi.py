# !pip install -q git+https://github.com/openai/whisper.git > /dev/null
# !pip install -q git+https://github.com/pyannote/pyannote-audio > /dev/null
path='/media/sanslab/Data/DuyLong/whis/audio.mp3'
num_speakers = 2 #@param {type:"integer"}

language = 'Vi' #@param ['any', 'English']

model_size = 'large' #@param ['tiny', 'base', 'small', 'medium', 'large']
from pydub import AudioSegment

def convert_to_mono(input_file, output_file):
  """Converts a stereo audio file to mono.

  Args:
    input_file: Path to the input audio file.
    output_file: Path to the output mono audio file.
  """

  sound = AudioSegment.from_file(input_file)
  sound = sound.set_channels(1)
  sound.export(output_file, format="wav")  # Or any other desired format

# Example usage:
input_file = "audio.wav"
output_file = "audio.wav"



model_name = 'large-v3'
# if language == 'Vi' and model_size != 'large':
#   model_name += '.vi'
  
# import whisper
import datetime

import subprocess

import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import whisper
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

if path[-3:] != 'wav':
  subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
  path = 'audio.wav'
convert_to_mono(input_file, output_file)
model = whisper.load_model(model_size)


result = model.transcribe(path)
# print(result)
segments = result["segments"]
# for text in segments:
#   print(text)
with contextlib.closing(wave.open(path,'r')) as f:
  frames = f.getnframes()
  rate = f.getframerate()
  duration = frames / float(rate)
  
audio = Audio()

def segment_embedding(segment):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)
  return embedding_model(waveform[None])

embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
  embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)
clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
  segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
                                            
def time(secs):
  return datetime.timedelta(seconds=round(secs))

f = open("transcript.txt", "w")

for (i, segment) in enumerate(segments):
  if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
    f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
  f.write(segment["text"][1:] + ' ')
f.close()


