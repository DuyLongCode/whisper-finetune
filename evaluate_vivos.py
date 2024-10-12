import torch
from transformers import pipeline
import datasets
import jiwer


import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
from IPython.display import Audio    
    
def calculate_wer(hypotheses, references):
  wer_score = jiwer.wer(references, hypotheses)
  return wer_score

def main():
  # Load dataset from Hugging Face
  dataset = datasets.load_dataset("quocanh34/viet_vivos",split='test')

  # Create a pipeline
  model_name = "openai/whisper-small"  # Replace with your desired model
  pipe = pipeline("automatic-speech-recognition", model=model_name)

  hypotheses = []
  references = []

  for example in dataset["test"]:
    audio_file = example["audio"]  # Assuming audio is in the 'audio' column
    reference_text = example["transcription"]  # Assuming transcription is in the 'transcription' column

    # Preprocess audio if necessary
    # ...

    # transcription = pipe(audio_file)["text"]
    transcription = transcribe(audio_file)['text']
    
    hypotheses.append(transcription)
    references.append(reference_text)

  wer = calculate_wer(hypotheses, references)
  print("WER:", wer)

if __name__ == "__main__":
  main()





# transcribe('/Users/duylong/Code/DoAn/denoise/recording.wav')
# iface = gr.Interface(
#     fn=transcribe, 
#     inputs=gr.Audio(sources="microphone", type="filepath"), 
#     outputs="text",
#     title="Whisper VietNam",
#     description="Realtime demo for VietNam speech recognition using a fine-tuned Whisper small model.",
# )

# iface.launch()