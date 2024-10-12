
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def llm_correction(prompt):
    # Configure the API

    rule = """You are an expert Vietnamese speech recognition assistant. Your task is to correct potential errors in speech transcriptions focusing on Vietnamese language nuances.
    You need to first consider the following variant sentences and try to pick corrected words from them: variant[0] … variant[n].
    Instructions:
    Pick the most confident sentence and then correct it.
    Correct words that seem incorrect and fit with VietNamese grammar
    Maintain the original sentence structure and word order.
    Do not change sentence structure and number of word and Do not ADD word
    Remove all dot from the final output.
    If the sentence is already correct and natural-sounding in Vietnamese leave it unchanged.
    Output only the corrected Vietnamese sentence without any explanations.
    Use only Vietnamese this is important do not use other language
    Correct sentence for medical domain
    Use accented Vietnamese and add space between each word
    DO NOT ADD PADDING
    """

    prompt_rule = 'These are the variants please pick and correct:\n'
    for i, pro in enumerate(prompt):
        prompt_rule += f'Variant {i}: {pro}\n'

    full_prompt = rule + "\n\n" + prompt_rule
    print(full_prompt)
  
    return full_prompt

# Example usage
if __name__ == "__main__":
    variants = ["Xin chào thế giới", "Sin chào thế giơi", "Xin chào thế giớ"]
    result = llm_correction(variants)
    print(f"Corrected sentence: {result}")
