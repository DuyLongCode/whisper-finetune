import genai
from genai.extensions.langchain import LangChainInterface
from genai.schema import HarmCategory, HarmBlockThreshold

def llm_correction(prompt):
    generation_config = {
        "temperature": 1,
        "top_p": 0.1,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
        }
    )
    
    rule = """You are an expert Vietnamese speech recognition assistant. Your task is to correct potential errors in speech transcriptions, focusing on Vietnamese language nuances.
    You need to first consider the following variant sentences and try to pick corrected words from them: variant[0] … variant[n].
    Instructions:
    Pick the most confident sentence and then correct it.
    Correct words that seem incorrect and fit with VietNamese grammar
    Maintain the original sentence structure and word order.
    Do not change sentence structure and number of word and Do not ADD word
    Remove all dot from the final output.
    If the sentence is already correct and natural-sounding in Vietnamese, leave it unchanged.
    Output only the corrected Vietnamese sentence without any explanations.
    Use only Vietnamese this is important, do not use other language
    Correct sentence for medical domain
    Use accented Vietnamese and add space between each word
    DO NOT ADD PADDING
    
    After providing the corrected sentence, on a new line, provide a confidence score between 0 and 1, where 1 is highest confidence and 0 is lowest confidence. Base this score on how certain you are about the corrections made.
    """
    
    prompt_rule = 'These are the variants, please pick and correct:\n'
    for i, pro in enumerate(prompt):
        prompt_rule += f'Variant {i}: {pro}\n'
    full_prompt = rule + "\n\n" + prompt_rule
    print(full_prompt)
    
    try:
        response = model.generate_content(full_prompt)
        response_text = response.text.strip()
        
        # Split the response into corrected text and confidence score
        lines = response_text.split('\n')
        if len(lines) >= 2:
            corrected_text = lines[0].strip()
            try:
                confidence_score = float(lines[1].strip())
            except ValueError:
                confidence_score = 0.5  # Default score if parsing fails
        else:
            corrected_text = response_text
            confidence_score = 0.5  # Default score if format is unexpected
        
        return corrected_text, confidence_score
    except Exception as e:
        print(f"Error in API call: {e}")
        return None, 0.0

# Example usage
prompt = ["Đây là một câu ví dụ", "Đây là một câu vi du"]
corrected_text, confidence = llm_correction(prompt)
print(f"Corrected text: {corrected_text}")
print(f"Confidence score: {confidence}")