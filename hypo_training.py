import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper.generation_whisper", message="The input name `inputs` is deprecated")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model='vinai/PhoWhisper-medium'



pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    torch_dtype=torch_dtype,
    device=device,
)



from gemini_training import *


paths=[]
texts=[]
with open('/media/sanslab/Data/DuyLong/train.txt','r+') as f:
    for line in f:
        print(line)
        paths.append(line.split('|')[0].rstrip('\n'))
        texts.append(line.split('|')[1])
        
print(paths)
from loadDataset import load_txt

from jiwer import wer, cer
import numpy as np


original_texts = []
without_correct_texts = []
corrected_texts = []
count = 0

NUMBER_OF_SAMPLE=5
TEMPERATURE=0.6
import time
from ratelimit import limits, sleep_and_retry

# Define the rate limit: 15 calls per 60 seconds
CALLS = 15
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def rate_limited_request(pipe, path):
    return pipe(path)
import string

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Example usage

import csv 
with open(f'/media/sanslab/Data/DuyLong/whis/llmtrain_{NUMBER_OF_SAMPLE}_{TEMPERATURE}.csv','w+',encoding='utf-8') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['Input', 'Output'])
    for path, text in zip(paths, texts):
        results = [None] * NUMBER_OF_SAMPLE
        for i in range(NUMBER_OF_SAMPLE):
            results[i] = remove_punctuation(pipe(path, generate_kwargs={"temperature": TEMPERATURE})['text'])
        corrected_text =rate_limited_request(llm_correction,results)
        csvwriter.writerow([corrected_text,text])

