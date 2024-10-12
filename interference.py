from transformers import pipeline,AutoModelForSpeechSeq2Seq,AutoProcessor

import gradio as gr
from correction import llm_correction
# name='vinai/PhoWhisper-medium'
model='vinai/PhoWhisper-medium'
audio='/media/sanslab/Data/DuyLong/vivos/VIVOSDEV01_R028.wav'
pipe = pipeline("automatic-speech-recognition",model=model,device='cuda')  # change to "your-username/the-name-you-picked"
text = pipe(audio)
print(text)