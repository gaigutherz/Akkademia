prediction_length = 100

import sys, os, datetime
import json
import torch
import random
import glob
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import DatasetDict, Dataset
import accelerate

import re
import requests
import unicodedata

import os


import logging
logger = logging.getLogger('pytorch_lightning.utilities.rank_zero')
class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        return 'available:' not in record.getMessage()

logger.addFilter(IgnorePLFilter())

source_langs = set(["akk"])
target_langs = set(["en"])

def get_finetune_model_id(model_id):
    model_dir = f"../results/{model_id}"
    checkpoints = [(os.path.abspath(x), int(os.path.split(x)[1].split("-")[1])) for x in glob.glob(f"{model_dir}/checkpoint-*")]
    checkpoints = sorted(checkpoints, key=lambda x: x[1])[-1]
    return checkpoints[0]

#os.environ["WANDB_NOTEBOOK_NAME"] = "TrainTranslator.ipynb"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

base_model_id = "UBC-NLP/AraT5v2-base-1024"
finetune_model_id = None
# finetune_model_id = get_finetune_model_id("t5-base-p-akksux-en-20220722-173018")

#model_max_length = 512
#batch_size = 8 if os.path.basename(base_model_id).startswith("t5-base") else 128

batch_size=1

num_train_epochs = 10

is_bi = False
use_paragraphs = True
use_lines = True
is_finetune = finetune_model_id is not None and len(finetune_model_id) > 1

date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
flags = ""
suffix = ""
if is_bi:
    flags += "-bi"

if use_paragraphs:
    flags += "-p"

if use_lines:
    flags += "-l"

if is_finetune:
    flags += "-f"
    suffix += f"-{os.path.basename(os.path.split(finetune_model_id)[0])}-{os.path.basename(finetune_model_id)}"

model_id = f"{os.path.basename(base_model_id)}{flags}-{''.join(sorted(list(source_langs)))}-{''.join(sorted(list(target_langs)))}-{date_id}{suffix}"
model_id


device = torch.device("cuda" if torch.cuda.is_available() else "mps")



# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString_en(s, use_prefix=False, task="Translate", target="cuneiform", characters="simple"):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    s = s.strip()
    if use_prefix:
        if task=="Translate":
            if target=="cuneiform":
                return 'Translate English to Akkadian cuneiform: ' + s
            elif target=="transliteration":
                if characters == "simple":
                    return 'Translate English to simple Akkadian transliteration: ' + s
                elif characters == "group":
                    return 'Translate English to grouped Akkadian transliteration: ' + s
    else:
        return s


# Lowercase, trim, and remove non-letter characters
def normalizeString_akk_transliterate(s, use_prefix=True):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    normalized_string = s.strip()
    if use_prefix:
        return 'Transliterate Akkadian cuneiform to Latin characters: ' + normalized_string
    else:
        return normalized_string

# Lowercase, trim, and remove non-letter characters
def normalizeString_akk_rev_transliterate(s, use_prefix=True):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    normalized_string = s.strip()
    if use_prefix:
        return 'Convert simple transliterated Latin characters to Akkadian cuneiform: ' + normalized_string
    else:
        return normalized_string

# Lowercase, trim, and remove non-letter characters
def normalizeString_akk_transliterate_translate(s, use_prefix=True, task="Translate"):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    normalized_string = s.strip()
    if use_prefix:
        if task == "Translate":
            return 'Translate Akkadian simple transliteration to English: ' + normalized_string
        elif task == "Group":
            return 'Group Akkadian transliteration into likely words: ' + normalized_string
    else:
        return normalized_string

# Lowercase, trim, and remove non-letter characters
def normalizeString_akk_transliterate_minimal(s, use_prefix=True):
    s = unicodeToAscii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    normalized_string = s.strip()
    if use_prefix:
        return 'Translate Akkadian grouped transliteration to English: ' + normalized_string
    else:
        return normalized_string

def normalizeString_akk(s, use_prefix=True, task="Translate", characters="simple"):
    # Optional: Remove unwanted modern characters, if any (adjust regex as needed)
    # s = re.sub(r'[^\u12000-\u123FF\u12400-\u1247F]+', '', s)  # Adjust Unicode ranges to cuneiform and related characters
    # Split each character/sign into separate entries
    # This assumes each character in the string is a distinct sign, no need to join with spaces if already separated
    normalized_string = ' '.join(s)  # This joins every character with a space, treating each as a separate token
    # Add the prefix if use_prefix is True
    if use_prefix:
        if task == "Translate":
            return 'Translate Akkadian cuneiform to English: ' + normalized_string
        elif task == "Transliterate":
            if characters == "simple":
                return 'Transliterate Akkadian cuneiform to simple Latin characters: ' + normalized_string
            elif characters == "group":
                return 'Transliterate Akkadian cuneiform to grouped Latin characters: ' + normalized_string
    else:
        return normalized_string

def read_and_process_file(file_path):
    # Check if the file_path is a URL
    if file_path.startswith('http://') or file_path.startswith('https://'):
        # Fetch the content from the URL
        response = requests.get(file_path)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        lines = response.text.strip().split('\n')
    else:
        # Open the local file and read the lines
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().strip().split('\n')
    # Replace ". . . " with "*" in each line
    processed_lines = [re.sub(r'\s*\.\s*\.\s*\.\s*', '*', line) for line in lines]
    return processed_lines

def convert(lst):
   res_dict = {}
   for i in range(0, len(lst), 2):
       res_dict[lst[i]] = lst[i + 1]
   return res_dict

def trim_pairs(pairs, max_length1, max_length2, length_threshold):
    # Filter out pairs where either element exceeds the word count limit
    filtered_pairs = [
        pair for pair in pairs
        if len(pair[0].split()) <= length_threshold and len(pair[1].split()) <= length_threshold
    ]
    # Trim each element in the pair to the maximum character length
    trimmed_pairs = [
        (s1[:max_length1], s2[:max_length2]) for s1, s2 in filtered_pairs
    ]
    return trimmed_pairs


def readLangs_cuneiform_translate(max_length_akk=5000, max_length_en=5000, length_threshold=100):
    print("Reading lines...")
    # Read the file and split into lines
    akk_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.ak')
    transcription_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.tr')
    en_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.en')
    akk_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.ak')
    transcription_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.tr')
    en_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.en')
    akk_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.ak')
    transcription_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.tr')
    en_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.en')
    akk = []
    for dataset in [akk_train, akk_test, akk_val]:
        for example in dataset:
                akk.append(example)
    en = []
    for dataset in [en_train, en_test, en_val]:
        for example in dataset:
                en.append(example)
    # Split every line into pairs and normalize
    train_pairs = [[normalizeString_akk(akk_train[l]), normalizeString_en(en_train[l])] for l in range(len(akk_train))]
    test_pairs = [[normalizeString_akk(akk_test[l]), normalizeString_en(en_test[l])] for l in range(len(akk_test))]
    val_pairs = [[normalizeString_akk(akk_val[l]), normalizeString_en(en_val[l])] for l in range(len(akk_val))]
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs imported: {len(pairs)}")
    #train_pairs = [[normalizeString_akk_transliterate(transcription_train[l]), normalizeString_en(en_train[l])] for l in range(len(akk_train))]
    #test_pairs = [[normalizeString_akk_transliterate(transcription_test[l]), normalizeString_en(en_test[l])] for l in range(len(akk_test))]
    #val_pairs = [[normalizeString_akk_transliterate(transcription_valid[l]), normalizeString_en(en_valid[l])] for l in range(len(akk_valid))]
    train_pairs = trim_pairs(train_pairs, max_length_akk, max_length_en, length_threshold)
    val_pairs = trim_pairs(val_pairs, max_length_akk, max_length_en, length_threshold)
    test_pairs = trim_pairs(test_pairs, max_length_akk, max_length_en, length_threshold)
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0]))
    max_length_pair_1 = max(pairs, key=lambda pair: len(pair[1]))
    print("Largest entry in pair[0]:")
    print(f"Length: {len(max_length_pair_0[0])}, Content: {max_length_pair_0[0]}")
    print("Largest entry in pair[1]:")
    print(f"Length: {len(max_length_pair_1[1])}, Content: {max_length_pair_1[1]}")
    mean_length_pair_0 = sum(len(pair[0]) for pair in pairs) / len(pairs)
    mean_length_pair_1 = sum(len(pair[1]) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in Akkadian: {mean_length_pair_0:.2f}")
    print(f"Mean number of tokens in English: {mean_length_pair_1:.2f}")
    return train_pairs, val_pairs, test_pairs, pairs

# Read your data
max_length = prediction_length
cuneiform_pad = max_length
cuneiform_to_english_pad = int(max_length*0.62)
transliterated_to_english_pad = int(max_length*0.71)
train_pairs, val_pairs, test_pairs, pairs = readLangs_cuneiform_translate(5000, 5000, max_length)

# Function to calculate the average number of tokens for sentence pairs
def calculate_average_tokens(pairs):
    total_tokens_lang_a = 0
    total_tokens_lang_b = 0
    for sentence_a, sentence_b in pairs:
        tokens_a = sentence_a.split()
        tokens_b = sentence_b.split()
        total_tokens_lang_a += len(tokens_a)
        total_tokens_lang_b += len(tokens_b)
    avg_tokens_lang_a = total_tokens_lang_a / len(pairs)
    avg_tokens_lang_b = total_tokens_lang_b / len(pairs)
    return avg_tokens_lang_a, avg_tokens_lang_b

avg_tokens_a, avg_tokens_b = calculate_average_tokens(pairs)
print(f"Average tokens in Akkadian: {avg_tokens_a}")
print(f"Average tokens in English: {avg_tokens_b}")

# Create data dictionaries
train_data_dict = {"akk": [pair[0] for pair in train_pairs], "en": [pair[1] for pair in train_pairs]}
val_data_dict = {"akk": [pair[0] for pair in val_pairs], "en": [pair[1] for pair in val_pairs]}
test_data_dict = {"akk": [pair[0] for pair in test_pairs], "en": [pair[1] for pair in test_pairs]}

# Create datasets
train_dataset = Dataset.from_dict(train_data_dict)
val_dataset = Dataset.from_dict(val_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)

translations = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})

tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(finetune_model_id if is_finetune else base_model_id)
#model_path = "<model_path>"
#tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


# Function to tokenize the examples
def tokenize_function(example):
    tokenized_inputs = tokenizer(example['akk'], padding="max_length", truncation=True, max_length=max_length)
    with tokenizer.as_target_tokenizer():
        tokenized_targets = tokenizer(example['en'], padding="max_length", truncation=True, max_length=max_length)
    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    return tokenized_inputs

# Tokenize the datasets
tokenized_datasets = DatasetDict({
    "train": translations["train"].map(tokenize_function, batched=True),
    "val": translations["val"].map(tokenize_function, batched=True),
    "test": translations["test"].map(tokenize_function, batched=True),
})



# Define the data collator
import numpy as np
from datasets import load_metric
metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Ensure all token IDs are valid
    valid_token_ids = set(tokenizer.get_vocab().values())
    preds = np.where(np.isin(preds, list(valid_token_ids)), preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Define the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

print("saving every ", int(len(train_pairs)/batch_size))
# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=model_path,
    evaluation_strategy="steps",
    learning_rate=2*2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    fp16=False,
    save_steps=500,
    eval_steps=500,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    predict_with_generate = True,
    #use_mps_device=True,
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

#subset = tokenized_datasets["val"].select(range(10))

test_preds = trainer.predict(tokenized_datasets["val"], max_length=prediction_length)
print(test_preds[2])

from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd

def normalize_spaces(text):
    # Replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()

def translate_cuneiform_set(tokenized_datasets, set='val'):
    results = []  # Initialize a list to hold results
    for test_number in tqdm(range(len(tokenized_datasets[set])), desc="Translating"):
        try:
            original = tokenized_datasets[set][test_number]['akk']
            reference = tokenized_datasets[set][test_number]['en']
            if len(reference) < 2:  # Check if the reference length is too short
                continue  # Skip this iteration if reference is too short
            pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda", max_length=max(20, len(reference)))
            predicted_texts = pipe(original)
            # Extract generated text and normalize spaces
            translation_texts = [item['generated_text'] for item in predicted_texts if 'generated_text' in item]
            normalized_predicted_text = ' '.join([normalize_spaces(text) for text in translation_texts])
            # Append the original, reference, and predicted text to the results list
            results.append({"Cuneiform": original, "Reference": reference, "Predicted": normalized_predicted_text})
        except Exception as e:  # General exception to catch unexpected issues
            print(f"Error processing entry {test_number}: {e}")
    # Convert the list of dictionaries into a DataFrame
    result_df = pd.DataFrame(results)
    # Define the full path for the CSV file
    csv_file_path = os.path.join(model_path, f'translated_cuneiform_{set}_set.csv')
    # Save the DataFrame to a CSV file in the specified directory
    result_df.to_csv(csv_file_path, index=False)
    return result_df


translated_results_val = translate_cuneiform_set(tokenized_datasets, 'val')
#translated_results_test = translate_cuneiform_set(tokenized_datasets, 'test')
#translated_results_train = translate_cuneiform_set(tokenized_datasets, 'train')


#####Transliterated translation
def readLangs_cuneiform_transliterated_translate(max_length_akk=5000, max_length_en=5000, length_threshold=100):
    print("Reading lines...")
    # Read the file and split into lines
    akk_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.ak')
    transcription_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.tr')
    en_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.en')
    akk_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.ak')
    transcription_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.tr')
    en_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.en')
    akk_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.ak')
    transcription_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.tr')
    en_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.en')
    akk = []
    for dataset in [akk_train, akk_test, akk_val]:
        for example in dataset:
                akk.append(example)
    en = []
    for dataset in [en_train, en_test, en_val]:
        for example in dataset:
                en.append(example)
        ###Translate from simple transliterated Akkadian to English
    train_pairs = [[normalizeString_akk_transliterate_translate(transcription_train[l], use_prefix=True, task="Translate"), normalizeString_en(en_train[l], use_prefix=False)] for l in range(len(transcription_train))]
    test_pairs = [[normalizeString_akk_transliterate_translate(transcription_test[l], use_prefix=True, task="Translate"), normalizeString_en(en_test[l], use_prefix=False)] for l in range(len(transcription_test))]
    val_pairs = [[normalizeString_akk_transliterate_translate(transcription_val[l], use_prefix=True, task="Translate"), normalizeString_en(en_val[l], use_prefix=False)] for l in range(len(transcription_val))]
    train_pairs = trim_pairs(train_pairs, max_length_akk, max_length_en, length_threshold)
    val_pairs = trim_pairs(val_pairs, max_length_akk, max_length_en, length_threshold)
    test_pairs = trim_pairs(test_pairs, max_length_akk, max_length_en, length_threshold)
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0]))
    max_length_pair_1 = max(pairs, key=lambda pair: len(pair[1]))
    print("Largest entry in pair[0]:")
    print(f"Length: {len(max_length_pair_0[0])}, Content: {max_length_pair_0[0]}")
    print("Largest entry in pair[1]:")
    print(f"Length: {len(max_length_pair_1[1])}, Content: {max_length_pair_1[1]}")
    mean_length_pair_0 = sum(len(pair[0]) for pair in pairs) / len(pairs)
    mean_length_pair_1 = sum(len(pair[1]) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in Akkadian: {mean_length_pair_0:.2f}")
    print(f"Mean number of tokens in English: {mean_length_pair_1:.2f}")
    return train_pairs, val_pairs, test_pairs, pairs

# Read your data
max_length = prediction_length
cuneiform_pad = max_length
cuneiform_to_english_pad = int(max_length*0.62)
transliterated_to_english_pad = int(max_length*0.71)
train_pairs, val_pairs, test_pairs, pairs = readLangs_cuneiform_transliterated_translate(5000, 5000, max_length)


# Create data dictionaries
train_data_dict = {"akk": [pair[0] for pair in train_pairs], "en": [pair[1] for pair in train_pairs]}
val_data_dict = {"akk": [pair[0] for pair in val_pairs], "en": [pair[1] for pair in val_pairs]}
test_data_dict = {"akk": [pair[0] for pair in test_pairs], "en": [pair[1] for pair in test_pairs]}

# Create datasets
train_dataset = Dataset.from_dict(train_data_dict)
val_dataset = Dataset.from_dict(val_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)

translations = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})

tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(finetune_model_id if is_finetune else base_model_id)
#model_path = "<model_path>"
#tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


# Function to tokenize the examples
def tokenize_function(example):
    tokenized_inputs = tokenizer(example['akk'], padding="max_length", truncation=True, max_length=max_length)
    with tokenizer.as_target_tokenizer():
        tokenized_targets = tokenizer(example['en'], padding="max_length", truncation=True, max_length=max_length)
    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    return tokenized_inputs

# Tokenize the datasets
tokenized_datasets = DatasetDict({
    "train": translations["train"].map(tokenize_function, batched=True),
    "val": translations["val"].map(tokenize_function, batched=True),
    "test": translations["test"].map(tokenize_function, batched=True),
})


# Define the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


print("saving every ", int(len(train_pairs)/batch_size))
# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=model_path,
    evaluation_strategy="steps",
    learning_rate=2*2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    predict_with_generate = True,
    #use_mps_device=True,
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

test_preds_transliterated_translate = trainer.predict(tokenized_datasets["val"], max_length=prediction_length)
print(test_preds_transliterated_translate[2])


from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd

def normalize_spaces(text):
    # Replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()

def translate_transliterated_set(tokenized_datasets, set='val'):
    results = []  # Initialize a list to hold results
    for test_number in tqdm(range(len(tokenized_datasets[set])), desc="Translating"):
        try:
            original = tokenized_datasets[set][test_number]['akk']
            reference = tokenized_datasets[set][test_number]['en']
            if len(reference) < 2:  # Check if the reference length is too short
                continue  # Skip this iteration if reference is too short
            pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda", max_length=max(20, len(reference)))
            predicted_texts = pipe(original)
            # Extract generated text and normalize spaces
            translation_texts = [item['generated_text'] for item in predicted_texts if 'generated_text' in item]
            normalized_predicted_text = ' '.join([normalize_spaces(text) for text in translation_texts])
            # Append the original, reference, and predicted text to the results list
            results.append({"Cuneiform": original, "Reference": reference, "Predicted": normalized_predicted_text})
        except Exception as e:  # General exception to catch unexpected issues
            print(f"Error processing entry {test_number}: {e}")
    # Convert the list of dictionaries into a DataFrame
    result_df = pd.DataFrame(results)
    # Define the full path for the CSV file
    csv_file_path = os.path.join(model_path, f'translated_transliterated_{set}_set.csv')
    # Save the DataFrame to a CSV file in the specified directory
    result_df.to_csv(csv_file_path, index=False)
    return result_df


translated_transliterated_results_val = translate_transliterated_set(tokenized_datasets, 'val')
#translated_transliterated_results_test = translate_transliterated_set(tokenized_datasets, 'test')
#translated_transliterated_results_train = translate_transliterated_set(tokenized_datasets, 'train')



#####Transliteraton
def readLangs_cuneiform_transliterate(max_length_akk=5000, max_length_en=5000, length_threshold=100):
    print("Reading lines...")
    # Read the file and split into lines
    akk_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.ak')
    transcription_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.tr')
    en_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.en')
    akk_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.ak')
    transcription_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.tr')
    en_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.en')
    akk_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.ak')
    transcription_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.tr')
    en_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.en')
    akk = []
    for dataset in [akk_train, akk_test, akk_val]:
        for example in dataset:
                akk.append(example)
    en = []
    for dataset in [en_train, en_test, en_val]:
        for example in dataset:
                en.append(example)
    ###Transliterate from Akkadian Cuenfirom to Latin characters
    train_pairs = [[normalizeString_akk(akk_train[l], use_prefix=True, task="Transliterate"), normalizeString_akk_transliterate(transcription_train[l], use_prefix=False)] for l in range(len(akk_train))]
    test_pairs = [[normalizeString_akk(akk_test[l], use_prefix=True, task="Transliterate"), normalizeString_akk_transliterate(transcription_test[l], use_prefix=False)] for l in range(len(akk_test))]
    val_pairs = [[normalizeString_akk(akk_val[l], use_prefix=True, task="Transliterate"), normalizeString_akk_transliterate(transcription_val[l], use_prefix=False)] for l in range(len(akk_val))]
    train_pairs = trim_pairs(train_pairs, max_length_akk, max_length_en, length_threshold)
    val_pairs = trim_pairs(val_pairs, max_length_akk, max_length_en, length_threshold)
    test_pairs = trim_pairs(test_pairs, max_length_akk, max_length_en, length_threshold)
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0]))
    max_length_pair_1 = max(pairs, key=lambda pair: len(pair[1]))
    print("Largest entry in pair[0]:")
    print(f"Length: {len(max_length_pair_0[0])}, Content: {max_length_pair_0[0]}")
    print("Largest entry in pair[1]:")
    print(f"Length: {len(max_length_pair_1[1])}, Content: {max_length_pair_1[1]}")
    mean_length_pair_0 = sum(len(pair[0]) for pair in pairs) / len(pairs)
    mean_length_pair_1 = sum(len(pair[1]) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in Akkadian: {mean_length_pair_0:.2f}")
    print(f"Mean number of tokens in English: {mean_length_pair_1:.2f}")
    return train_pairs, val_pairs, test_pairs, pairs

# Read your data
max_length = prediction_length
cuneiform_pad = max_length
cuneiform_to_english_pad = int(max_length*0.62)
transliterated_to_english_pad = int(max_length*0.71)
train_pairs, val_pairs, test_pairs, pairs = readLangs_cuneiform_transliterate(5000, 5000, max_length)


# Create data dictionaries
train_data_dict = {"akk": [pair[0] for pair in train_pairs], "en": [pair[1] for pair in train_pairs]}
val_data_dict = {"akk": [pair[0] for pair in val_pairs], "en": [pair[1] for pair in val_pairs]}
test_data_dict = {"akk": [pair[0] for pair in test_pairs], "en": [pair[1] for pair in test_pairs]}

# Create datasets
train_dataset = Dataset.from_dict(train_data_dict)
val_dataset = Dataset.from_dict(val_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)

translations = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})

tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(finetune_model_id if is_finetune else base_model_id)
#model_path = "<model_path>"
#tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


# Function to tokenize the examples
def tokenize_function(example):
    tokenized_inputs = tokenizer(example['akk'], padding="max_length", truncation=True, max_length=max_length)
    with tokenizer.as_target_tokenizer():
        tokenized_targets = tokenizer(example['en'], padding="max_length", truncation=True, max_length=max_length)
    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    return tokenized_inputs

# Tokenize the datasets
tokenized_datasets = DatasetDict({
    "train": translations["train"].map(tokenize_function, batched=True),
    "val": translations["val"].map(tokenize_function, batched=True),
    "test": translations["test"].map(tokenize_function, batched=True),
})


# Define the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


print("saving every ", int(len(train_pairs)/batch_size))
# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=model_path,
    evaluation_strategy="steps",
    learning_rate=2*2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    predict_with_generate = True,
    #use_mps_device=True,
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

test_preds_transliterate = trainer.predict(tokenized_datasets["val"], max_length=prediction_length)
print(test_preds_transliterate[2])

from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd

def normalize_spaces(text):
    # Replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()

def transliterate_cuneiform_set(tokenized_datasets, set='val'):
    results = []  # Initialize a list to hold results
    for test_number in tqdm(range(len(tokenized_datasets[set])), desc="Transliterating"):
        try:
            original = tokenized_datasets[set][test_number]['akk']
            reference = tokenized_datasets[set][test_number]['en']
            if len(reference) < 2:  # Check if the reference length is too short
                continue  # Skip this iteration if reference is too short
            pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda", max_length=max(20, len(reference)))
            predicted_texts = pipe(original)
            # Extract generated text and normalize spaces
            translation_texts = [item['generated_text'] for item in predicted_texts if 'generated_text' in item]
            normalized_predicted_text = ' '.join([normalize_spaces(text) for text in translation_texts])
            # Append the original, reference, and predicted text to the results list
            results.append({"Cuneiform": original, "Reference": reference, "Predicted": normalized_predicted_text})
        except Exception as e:  # General exception to catch unexpected issues
            print(f"Error processing entry {test_number}: {e}")
    # Convert the list of dictionaries into a DataFrame
    result_df = pd.DataFrame(results)
    # Define the full path for the CSV file
    csv_file_path = os.path.join(model_path, f'transliterated_cuneiform_{set}_set.csv')
    # Save the DataFrame to a CSV file in the specified directory
    result_df.to_csv(csv_file_path, index=False)
    return result_df


transliterated_results_val = transliterate_cuneiform_set(tokenized_datasets, 'val')
transliterated_results_test = transliterate_cuneiform_set(tokenized_datasets, 'test')
#transliterated_results_train = transliterate_cuneiform_set(tokenized_datasets, 'train')

df_val = pd.read_csv('~/GitHub/results/t5-small-p-l-akk-en-20240907-200329/transliterated_cuneiform_val_set.csv')  # Update the path to where your file is located
# Calculate accuracy by comparing the 'Reference' and 'Predicted' columns
val_accuracy = (df_val['Reference'] == df_val['Predicted']).mean()
print(f"Validation Accuracy: {val_accuracy:.4f}")

df_test = pd.read_csv('~/GitHub/results/t5-small-p-l-akk-en-20240907-200329/transliterated_cuneiform_test_set.csv')  # Update the path to where your file is located
# Calculate accuracy by comparing the 'Reference' and 'Predicted' columns
test_accuracy = (df_test['Reference'] == df_test['Predicted']).mean()
print(f"Test Accuracy: {test_accuracy:.4f}")

def calculate_token_accuracy(row):
    # Split the strings into tokens. Assuming space is the delimiter.
    reference_tokens = row['Reference'][0].split()
    predicted_tokens = row['Predicted'][0].split()
    # Ensure that both token lists are the same length by padding or truncating (optional)
    min_length = min(len(reference_tokens), len(predicted_tokens))
    reference_tokens = reference_tokens[:min_length]
    predicted_tokens = predicted_tokens[:min_length]
    # Calculate accuracy per token
    correct_tokens = sum(1 for ref, pred in zip(reference_tokens, predicted_tokens) if ref == pred)
    if min_length > 0:
        return correct_tokens / min_length
    else:
        return 0  # Avoid division by zero if there are no tokens

# Apply the function to each row and calculate the mean accuracy across all rows
df_val['Token Accuracy'] = df_val.apply(calculate_token_accuracy, axis=1)
overall_token_accuracy = df_val['Token Accuracy'].mean()
print(f"Token-level Validation Accuracy: {overall_token_accuracy:.4f}")


# Apply the function to each row and calculate the mean accuracy across all rows
df_test['Token Accuracy'] = df_test.apply(calculate_token_accuracy, axis=1)
overall_token_accuracy = df_test['Token Accuracy'].mean()
print(f"Token-level Test Accuracy: {overall_token_accuracy:.4f}")

#####Group
def readLangs_cuneiform_group(max_length_akk=5000, max_length_en=5000, length_threshold=100):
    print("Reading lines...")
    # Read the file and split into lines
    akk_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.ak')
    transcription_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.tr')
    en_train = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/train.en')
    akk_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.ak')
    transcription_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.tr')
    en_test = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/test.en')
    akk_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.ak')
    transcription_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.tr')
    en_val = read_and_process_file('https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input/not_divided_by_three_dots/valid.en')
    akk = []
    for dataset in [akk_train, akk_test, akk_val]:
        for example in dataset:
                akk.append(example)
    en = []
    for dataset in [en_train, en_test, en_val]:
        for example in dataset:
                en.append(example)
    ###Group transliterated Akkadian into words
    train_pairs = [[normalizeString_akk_transliterate_translate(transcription_train[l], use_prefix=True, task="Group"), normalizeString_akk_transliterate_minimal(transcription_train[l], use_prefix=False)] for l in range(len(akk_train))]
    test_pairs = [[normalizeString_akk_transliterate_translate(transcription_test[l], use_prefix=True, task="Group"), normalizeString_akk_transliterate_minimal(transcription_test[l], use_prefix=False)] for l in range(len(akk_test))]
    val_pairs = [[normalizeString_akk_transliterate_translate(transcription_val[l], use_prefix=True, task="Group"), normalizeString_akk_transliterate_minimal(transcription_val[l], use_prefix=False)] for l in range(len(akk_val))]
    train_pairs = trim_pairs(train_pairs, max_length_akk, max_length_en, length_threshold)
    val_pairs = trim_pairs(val_pairs, max_length_akk, max_length_en, length_threshold)
    test_pairs = trim_pairs(test_pairs, max_length_akk, max_length_en, length_threshold)
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0]))
    max_length_pair_1 = max(pairs, key=lambda pair: len(pair[1]))
    print("Largest entry in pair[0]:")
    print(f"Length: {len(max_length_pair_0[0])}, Content: {max_length_pair_0[0]}")
    print("Largest entry in pair[1]:")
    print(f"Length: {len(max_length_pair_1[1])}, Content: {max_length_pair_1[1]}")
    mean_length_pair_0 = sum(len(pair[0]) for pair in pairs) / len(pairs)
    mean_length_pair_1 = sum(len(pair[1]) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in Akkadian: {mean_length_pair_0:.2f}")
    print(f"Mean number of tokens in English: {mean_length_pair_1:.2f}")
    return train_pairs, val_pairs, test_pairs, pairs

# Read your data
max_length = prediction_length
cuneiform_pad = max_length
cuneiform_to_english_pad = int(max_length*0.62)
transliterated_to_english_pad = int(max_length*0.71)
train_pairs, val_pairs, test_pairs, pairs = readLangs_cuneiform_group(5000, 5000, max_length)

# Create data dictionaries
train_data_dict = {"akk": [pair[0] for pair in train_pairs], "en": [pair[1] for pair in train_pairs]}
val_data_dict = {"akk": [pair[0] for pair in val_pairs], "en": [pair[1] for pair in val_pairs]}
test_data_dict = {"akk": [pair[0] for pair in test_pairs], "en": [pair[1] for pair in test_pairs]}

# Create datasets
train_dataset = Dataset.from_dict(train_data_dict)
val_dataset = Dataset.from_dict(val_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)

translations = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})

tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(finetune_model_id if is_finetune else base_model_id)
#model_path = "<model_path>"
#tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


# Function to tokenize the examples
def tokenize_function(example):
    tokenized_inputs = tokenizer(example['akk'], padding="max_length", truncation=True, max_length=max_length)
    with tokenizer.as_target_tokenizer():
        tokenized_targets = tokenizer(example['en'], padding="max_length", truncation=True, max_length=max_length)
    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    return tokenized_inputs

# Tokenize the datasets
tokenized_datasets = DatasetDict({
    "train": translations["train"].map(tokenize_function, batched=True),
    "val": translations["val"].map(tokenize_function, batched=True),
    "test": translations["test"].map(tokenize_function, batched=True),
})


# Define the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

print("saving every ", int(len(train_pairs)/batch_size))
# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=model_path,
    evaluation_strategy="steps",
    learning_rate=2*2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    predict_with_generate = True,
    #use_mps_device=True,
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


test_preds_group = trainer.predict(tokenized_datasets["val"], max_length=prediction_length)
print(test_preds_group[2])

from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd

def normalize_spaces(text):
    # Replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()

def group_transliterations_set(tokenized_datasets, set='val'):
    results = []  # Initialize a list to hold results
    for test_number in tqdm(range(len(tokenized_datasets[set])), desc="Translating"):
        try:
            original = tokenized_datasets[set][test_number]['akk']
            reference = tokenized_datasets[set][test_number]['en']
            if len(reference) < 2:  # Check if the reference length is too short
                continue  # Skip this iteration if reference is too short
            pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda", max_length=max(20, len(reference)))
            predicted_texts = pipe(original)
            # Extract generated text and normalize spaces
            translation_texts = [item['generated_text'] for item in predicted_texts if 'generated_text' in item]
            normalized_predicted_text = ' '.join([normalize_spaces(text) for text in translation_texts])
            # Append the original, reference, and predicted text to the results list
            results.append({"Cuneiform": original, "Reference": reference, "Predicted": normalized_predicted_text})
        except Exception as e:  # General exception to catch unexpected issues
            print(f"Error processing entry {test_number}: {e}")
    # Convert the list of dictionaries into a DataFrame
    result_df = pd.DataFrame(results)
    # Define the full path for the CSV file
    csv_file_path = os.path.join(model_path, f'grouped_transliterations_{set}_set.csv')
    # Save the DataFrame to a CSV file in the specified directory
    result_df.to_csv(csv_file_path, index=False)
    return result_df


group_results_val = group_transliterations_set(tokenized_datasets, 'val')
group_results_test = group_transliterations_set(tokenized_datasets, 'test')
#group_results_train = group_transliterations_set(tokenized_datasets, 'train')

df_val = pd.read_csv('~/GitHub/results/t5-small-p-l-akk-en-20240907-200329/grouped_transliterations_val_set.csv')  # Update the path to where your file is located
# Calculate accuracy by comparing the 'Reference' and 'Predicted' columns
val_accuracy = (df_val['Reference'] == df_val['Predicted']).mean()
print(f"Validation Accuracy: {val_accuracy:.4f}")

df_test = pd.read_csv('~/GitHub/results/t5-small-p-l-akk-en-20240907-200329/grouped_transliterations_test_set.csv')  # Update the path to where your file is located
# Calculate accuracy by comparing the 'Reference' and 'Predicted' columns
test_accuracy = (df_test['Reference'] == df_test['Predicted']).mean()
print(f"Test Accuracy: {test_accuracy:.4f}")
