from transformers import AutoModelForCausalLM, AutoTokenizer

lm_model = AutoModelForCausalLM.from_pretrained("vinai/phobert-base")
lm_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def apply_lm(text):
    inputs = lm_tokenizer(text, return_tensors="pt")
    outputs = lm_model.generate(**inputs, max_length=100)
    print('load ok')
    return lm_tokenizer.decode(outputs[0], skip_special_tokens=True)

from transformers import pipeline
import gradio as gr
name='DuyND/finetuneWhisper'
pipe = pipeline(model=name)  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]
    text=apply_lm(text)
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(sources='upload', type="filepath"), 
    outputs="text",
    title="Whisper VietNam",
    description="Realtime demo for VietNam speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()


# apply toàn gái xinh này con mô con nựa nhìn ngất ngây như tấy mày có tin là con gái xinh là do nguồn nước không rần rần rần hắn xinh nhân đẹp là gió bố mẹ hắn gió giọng họ nhà hắn gió cái gen liên quan đến nguồn nước bây giờ mi nói xem cái con đang đạp xe đạp đang trước mặt đây này đã có chồng chưa mày nhìn đi mông nó tụt như thế kia tàu cá mới mày là nó có chồng và một con rồi hư đưa đi

# not  toàn gái xinh này con mô con nựa nhìn ngất ngây như tấy mày có tin là con gái xinh là do nguồn nước không rần rần rần hắn xinh nhân đẹp là gió bố mẹ hắn gió giọng họ nhà hắn gió cái gen liên quan đến nguồn nước bây giờ mi nói xem cái con đang đạp xe đạp đang trước mặt đây này đã có chồng chưa mày nhìn đi mông nó tụt như thế kia tàu cá mới mày là nó có chồng và một con rồi hư đưa đi
