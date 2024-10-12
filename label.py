from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import gradio as gr
from correction import llm_correction
import os
import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
model = 'vinai/PhoWhisper-large'
folder_path = "/media/sanslab/Data/DuyLong/VietMed_unlabeled_1000h_segmented_8kHz_650_700"
output_file = '/media/sanslab/Data/DuyLong/whis/label.txt'

# Initialize the pipeline
pipe = pipeline("automatic-speech-recognition", model=model, device='cuda')

# Get all audio file paths
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
              if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.wav', '.mp3', '.flac'))]

# Function to process a single file
def process_file(path):
    try:
        text = pipe(path)['text']
        return f'{path}|{text}\n'
    except Exception as e:
        logging.error(f"Error processing {path}: {str(e)}")
        return None

# Process files and write results
with open(output_file, 'w', encoding='utf-8') as file:
    for result in tqdm.tqdm(map(process_file, file_paths), total=len(file_paths), desc="Processing files"):
        if result:
            file.write(result)

logging.info(f"Processing complete. Results written to {output_file}")