# import wandb
# wandb.init(project="finetunewhisper")
model_name="DuyND/finetuneWhisper"
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_name,cache_dir="/media/sanslab/Data/DuyLong/whis",)

from datasets import load_dataset, DatasetDict,load_from_disk

vivos_test=load_dataset("quocanh34/viet_vivos",cache_dir='./data/',download_mode='reuse_dataset_if_exists',split='test')
commonvoice_test=load_dataset("mozilla-foundation/common_voice_13_0",'vi',cache_dir='./test/',split='test',)
def cleanData(data):
    
    data=[item for item in data if item]
    return data
    

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
def prepare_commonvoice(batch):
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    
    return batch
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name,cache_dir="/media/sanslab/Data/DuyLong/whis")
model.generation_config.language = "vi"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None
from datasets import Dataset, DatasetDict, Audio



# dataset2['train']= dataset2['train'].cast_column("audio", Audio(sampling_rate=16000))
# dataset2['test']= dataset2['test'].cast_column("audio", Audio(sampling_rate=16000))
commonvoice_test=commonvoice_test.cast_column("audio", Audio(sampling_rate=16000))

                                 
# print('start process')

# # dataset2 =dataset2.map(prepare_dataset, remove_columns=["input_values","input_length"], num_proc=2,batch_size=4)
# dataset2['train']=dataset2['train'].map(prepare_dataset, num_proc=2,batch_size=4)
# dataset2['test']=dataset2['test'].map(prepare_dataset, num_proc=2,batch_size=4)
commonvoice_test=commonvoice_test.map(prepare_commonvoice, num_proc=2,batch_size=4)

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


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small2-vi",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
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
)
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    
)


predictions = trainer.predict(commonvoice_test)

# Evaluate performance
print(predictions.metrics)
# trainer.train() 
# trainer.save_model('./model_finetune')

trainer.push_to_hub('DuyND/finetuneWhisper')


# test_wer': 25.564738292011018,