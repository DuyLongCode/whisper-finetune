# coding: utf8
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# model_path = "vinai/PhoGPT-4B-Chat"  
model_path='/media/sanslab/Data/DuyLong/models--vinai--PhoGPT-4B-Chat/snapshots/116013fa63f8c4025739487e1cbff65b7375bbe2'

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
config.init_device = "cuda"
# config.attn_config['attn_impl'] = 'flash' # If installed: this will use either Flash Attention V1 or V2 depending on what is installed

model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True).to('cuda')
# If your GPU does not support bfloat16:
# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
model.eval()  

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
def phoGPT(prompt):
    PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"  

    # Some instruction examples
    # instruction = "Viết bài văn nghị luận xã hội về {topic}"
    # instruction = "Viết bản mô tả công việc cho vị trí {job_title}"
    # instruction = "Sửa lỗi chính tả:\n{sentence_or_paragraph}"
    # instruction = "Dựa vào văn bản sau đây:\n{text}\nHãy trả lời câu hỏi: {question}"
    # instruction = "Tóm tắt văn bản:\n{text}"

    # instruction = "Sửa lỗi chính tả:\nTriệt phá băng nhóm kướp ô tô, sử dụng \"vũ khí nóng\""
    rule=f"""Bạn là một trợ lý tuyệt vời cho hệ thống nhận dạng giọng nói. Nhiệm vụ của bạn là kiểm tra và sửa các lỗi tiềm ẩn trong phiên âm giọng nói.
    Trước tiên, bạn cần xem xét các câu biến thể sau đây và cố gắng chọn các từ được sửa từ chúng: y[0] … y[n].
    Các quy tắc bổ sung cho sửa đổi này:
    1. Nếu từ nào trong câu gốc có vẻ lạ hoặc không nhất quán thì thay bằng từ tương ứng trong các câu biến thể.
    2. Bạn không cần phải sửa đổi câu gốc nếu nó đã ổn.
    3. Giữ nguyên cấu trúc câu và trật tự từ.
    4. Chỉ thay từ trong câu gốc bằng từ trong câu biến thể. Đừng chỉ thêm hoặc xóa các từ.
    5. Cố gắng làm cho câu đã sửa có số từ bằng với câu gốc.
    6. Bỏ qua dấu câu.
    7. Sử dụng tiếng Việt
    8. Chỉ đưa ra một câu sửa đổi và không có lời giải thích.
    9. Sửa lại cho phù hợp với tiếng việt
    """
    prompt_rule='this variant, please pick and correct: \n'
    for (i,pro) in enumerate(prompt):
        prompt_rule+=f'Variant {i}:{pro}\n'
    instruction=rule+prompt_rule
    
    input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})
    print(input_prompt)
    input_ids = tokenizer(input_prompt, return_tensors="pt")  
    
    outputs = model.generate(  
        inputs=input_ids["input_ids"].to("cuda"),  
        attention_mask=input_ids["attention_mask"].to("cuda"),  
        do_sample=True,  
        temperature=1.0,  
        top_k=50,  
        top_p=0.9,  
        max_new_tokens=1024,  
        eos_token_id=tokenizer.eos_token_id,  
        pad_token_id=tokenizer.pad_token_id  
    )  

    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
    response = response.split("### Trả lời:")[1]
    return response


