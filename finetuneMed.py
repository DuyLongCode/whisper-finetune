import wandb
wandb.init(project="finetunewhisper-med")
model_name="vinai/PhoWhisper-medium"
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)


from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_name,cache_dir="/media/sanslab/Data/DuyLong/whis",)

from datasets import load_dataset, DatasetDict,load_from_disk




from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name,cache_dir="/media/sanslab/Data/DuyLong/whis")


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
   
    # audio = batch["audio"]['array']
    batch["input_features"] = feature_extractor(batch["audio"]['array'], sampling_rate=16000).input_features[0]


    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["labels"]).input_ids
    
    return batch


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name,cache_dir="/media/sanslab/Data/DuyLong/whis")
model.generation_config.language = "vi"
model.generation_config.task = "transcribe"
model.config.dropout = 0.2
model.config.attention_dropout = 0.2
model.config.activation_dropout = 0.2
model.generation_config.forced_decoder_ids = None
from datasets import Dataset, DatasetDict

from datasets import Audio


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[ Union[List[int], torch.Tensor],str]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
import evaluate

metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
from loadDataset import load_txt
vietMed= load_txt(paths=['/media/sanslab/Data/DuyLong/train.txt'])
# vietMedLarge=load_txt(paths=['/media/sanslab/Data/DuyLong/whis/label.txt','/media/sanslab/Data/DuyLong/train.txt'])
# vietMed=vietMed.map(prepare_dataset, num_proc=1,batch_size=1,load_from_cache_file=True,writer_batch_size=1)
# import math

# chunk_size = 1000 # Adjust based on your memory constraints
# num_chunks = math.ceil(len(vietMed) / chunk_size)

# for i in range(num_chunks):
#     start_idx = i * chunk_size
#     end_idx = min((i + 1) * chunk_size, len(vietMed))
    
#     chunk = vietMed.select(range(start_idx, end_idx))
  
#     processed_chunk = chunk.map(
#         prepare_dataset,
#         batch_size=16,
#         num_proc=2,
#         load_from_cache_file=True,
#         writer_batch_size=1,
#         cache_file_name='/media/sanslab/Data/DuyLong/temp/chunk_cache.arrow'
#     )
#     # Save the processed chunk
#     processed_chunk.save_to_disk(f"/media/sanslab/Data/DuyLong/whis/data/med/processed_chunk_{i}")
# from datasets import concatenate_datasets
# vietMed = None
# import shutil
# for i in range(num_chunks):
#     chunk = Dataset.load_from_disk(f"/media/sanslab/Data/DuyLong/whis/data/med/processed_chunk_{i}")
#     vietMed = concatenate_datasets([vietMed, chunk]) if vietMed else chunk
#     del chunk
#     shutil.rmtree(f"/media/sanslab/Data/DuyLong/whis/data/med/processed_chunk_{i}")
# vietMed=vietMed.save_to_disk('/media/sanslab/Data/DuyLong/vietMed')
vietMed = load_from_disk('/media/sanslab/Data/DuyLong/vietMed')
vietMedVal= load_txt(paths=['/media/sanslab/Data/DuyLong/test.txt'])
vietMedVal=vietMedVal.map(
        prepare_dataset,
        batch_size=16,
        num_proc=1,
        load_from_cache_file=True,
        writer_batch_size=1,
        cache_file_name='/media/sanslab/Data/DuyLong/temp/chunk_cache_val.arrow'
    )

model.gradient_checkpointing_enable()
from transformers import Seq2SeqTrainingArguments


from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
train_step=500
eval_step=100
save_checkpoint=train_step/2
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium-med",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # increase by 2x for every 2x decrease in batch size
    gradient_checkpointing=True,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=train_step,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=eval_step,
    eval_steps=eval_step,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    optim="adafactor",
    remove_unused_columns=False
)
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vietMed,
    eval_dataset=vietMedVal,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

    
trainer.train() 
trainer.save_model('./model_finetune_MED')


trainer.model.save_pretrained("./model_finetune_lora")