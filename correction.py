# API Request JSON Cell
from llamaapi import LlamaAPI
import json


# Replace 'Your_API_Token' with your actual API token
def llm_correction(text):
  llama = LlamaAPI('LL-nPDPNZLFsN7czxR7FmnB70saNx2mP1uRlkXgGcm0HZd8EUcyvNAXkDcH2n8l61sl')
  
  system="""You are an excellent assistant for speech recognition system. Your task is to check and correct potential errors in speech transcriptions.
  Please follow the following rules, and here is the sentence to work on: y[1].
  You need to first consider the following variant sentences and try to pick corrected words from them: y[2] … y[n].
  Additional rules for this modification:
  1. If any word in the original sentence looks weird or inconsistent, then replace it with a corresponding word from variant sentences.
  2. You don’t have to modify the original sentence if it already looks good.
  3. Keep the sentence structure and word order intact.
  4. Only replace words in the original sentence with ones from variant sentences. Do not simply add or delete words.
  5. Try to make the corrected sentence have the same number of words as the original sentence.
  6. Ignore punctuation.
  7. Use VietNamese
  8. Output only one modified sentence and no explanation."""

  model='llama3.1-405b'
  model2='llama2-7b'
  api_request_json = {
  "model": model,
  "messages": [{"role": "system", "content": system},
  {"role": "user", "content": text},] 
  }

    # Make your request and handle the response
  response = llama.run(api_request_json)
  print(response.text)
  
  return response.text


# import whisper
# model = whisper.load_model('large-v3',)
# result = model.transcribe('/media/sanslab/Data/DuyLong/whis/audio.wav')
# segments = result["segments"]

# # Process each segment with LLM
# for segment in segments:
#     segment['text'] = process_with_llm(segment['text'])
    


