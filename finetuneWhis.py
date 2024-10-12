# import wandb
# wandb.init(project="finetunewhisper")
model_name="vinai/PhoWhisper-medium"
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)




from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_name,cache_dir="/media/sanslab/Data/DuyLong/whis",)

from datasets import load_dataset, DatasetDict,load_from_disk




# # Load your datasets
dataset2=DatasetDict()
# dataset2 = load_dataset("tuanmanh28/VIVOS_CommonVoice_FOSD_Control_processed_dataset",cache_dir="./data/", download_mode="reuse_dataset_if_exists")
dataset2['train']=load_dataset('natmin322/28k_vietnamese_voice_augmented_of_VigBigData',cache_dir='/media/sanslab/Data/DuyLong/whis/data',download_mode='reuse_dataset_if_exists',split='train_1')

dataset2['test']=load_dataset('natmin322/28k_vietnamese_voice_augmented_of_VigBigData',cache_dir='/media/sanslab/Data/DuyLong/whis/data',download_mode='reuse_dataset_if_exists',split='test')
vivos_test=load_dataset("quocanh34/viet_vivos",cache_dir='/media/sanslab/Data/DuyLong/whis/data',download_mode='reuse_dataset_if_exists',split='test')

def cleanData(data):
    
    data=[item for item in data if item]
    return data
    

cleanData(dataset2['test'])
cleanData(dataset2['train'])
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name,cache_dir="/media/sanslab/Data/DuyLong/whis")


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
   
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    
    return batch

def prepare_vivos(batch):
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    
    return batch
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name,cache_dir="/media/sanslab/Data/DuyLong/whis")
model.generation_config.language = "vi"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None
from datasets import Dataset, DatasetDict


print('load data')

from datasets import Audio

dataset2['train']= dataset2['train'].cast_column("audio", Audio(sampling_rate=16000))
# dataset2['test']= dataset2['test'].cast_column("audio", Audio(sampling_rate=16000))
vivos_test=vivos_test.cast_column("audio", Audio(sampling_rate=16000))
                                 
print('start process')

dataset2['train']=dataset2['train'].map(prepare_dataset, num_proc=2,batch_size=2,with_rank=True,load_from_cache_file=True,)
# dataset2['test']=dataset2['test'].map(prepare_dataset, num_proc=2,batch_size=4)
vivos_test=vivos_test.map(prepare_vivos, num_proc=2,batch_size=2,with_rank=True,load_from_cache_file=True,)
print(dataset2)
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
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


vietMed= load_txt(paths=['/media/sanslab/Data/DuyLong/whis/label.txt','/media/sanslab/Data/DuyLong/train.txt'])
model.gradient_checkpointing_enable()
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium-med",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # increase by 2x for every 2x decrease in batch size
    gradient_checkpointing=True,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    optim="adafactor"
    
)
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vietMed,
    eval_dataset=vivos_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    
)

    
trainer.train() 
trainer.save_model('./model_finetune')

