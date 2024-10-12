import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper.generation_whisper", message="The input name `inputs` is deprecated")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model='DuyND/finetuneMed'



pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    torch_dtype=torch_dtype,
    device=device,
)


from gemini import *


paths=[]
texts=[]
with open('/Users/duylong/Code/DoAn/whis/test.txt','r+') as f:
    for line in f:
        paths.append(line.split('|')[0].rstrip('\n'))
        texts.append(line.split('|')[1])
        
from jiwer import wer, cer

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

with open(f'/Users/duylong/Code/DoAn/llmcorrect_{NUMBER_OF_SAMPLE}_{TEMPERATURE}.txt', 'w+') as f:
    for path, text in zip(paths, texts):
        results = [None] * NUMBER_OF_SAMPLE
        for i in range(NUMBER_OF_SAMPLE):
            try:
                results[i] = remove_punctuation(pipe(path, generate_kwargs={"temperature": TEMPERATURE})['text'])
            except Exception as e:
                pass
        try:
            text_without_correct = remove_punctuation(pipe(path, generate_kwargs={"temperature": 1.0})['text'])
            print(text_without_correct)
            # corrected_text = llm_correction(results)
            corrected_text =rate_limited_request(llm_correction,results)
            # corrected_text = phoGPT(results)
            original_texts.append(text)
            without_correct_texts.append(text_without_correct)
            corrected_texts.append(corrected_text)
            
            f.write(f'Without Correct: {text_without_correct}\n')
            f.write(f'Correct        : {corrected_text}\n')
            f.write(f'Original       : {text}\n')
            f.write('-' * 50 + '\n')
            print(f'Correct        : {corrected_text}\n')
            count += 1
            if count % 10 == 0:
                wer_without_correct = wer(original_texts, without_correct_texts)
                cer_without_correct = cer(original_texts, without_correct_texts)
                wer_corrected = wer(original_texts, corrected_texts)
                cer_corrected = cer(original_texts, corrected_texts)
               
                f.write(f'Intermediate Results (after {count} sentences):\n')
                f.write(f'WER without correct: {wer_without_correct*100:.2f}%\n')
                f.write(f'CER without correct: {cer_without_correct*100:.2f}%\n')
                f.write(f'WER with correct: {wer_corrected*100:.2f}%\n')
                f.write(f'CER with correct: {cer_corrected*100:.2f}%\n')
                f.write('-' * 50 + '\n')
        except Exception as e:
            pass
        # Calculate final WER and CER
        # final_wer_without_correct = wer(original_texts, without_correct_texts)
        # final_cer_without_correct = cer(original_texts, without_correct_texts)
        # final_wer_corrected = wer(original_texts, corrected_texts)
        # final_cer_corrected = cer(original_texts, corrected_texts)

        # f.write('\nFinal Results:\n')
        # f.write(f'Overall WER Without Correct: {final_wer_without_correct*100:.2f}%\n')
        # f.write(f'Overall CER Without Correct: {final_cer_without_correct*100:.2f}%\n')
        # f.write(f'Overall WER Corrected: {final_wer_corrected*100:.2f}%\n')
        # f.write(f'Overall CER Corrected: {final_cer_corrected*100:.2f}%\n')
    
    
        