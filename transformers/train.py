#PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

import sys, os, datetime
import json
import torch
import random
import glob
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import DatasetDict, Dataset, load_metric
import accelerate

import numpy as np

import re
import requests
import unicodedata

from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments, EarlyStoppingCallback, BertTokenizer,MT5ForConditionalGeneration
from transformers.data.data_collator import DataCollatorForSeq2Seq,default_data_collator
import pandas as pd
import math,os
import numpy as np
from tqdm import tqdm
import torch

import os
os.chdir("~/GitHub/cuneiform")


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

base_model_id = "thalesian/akk-111m"
finetune_model_id = None
# finetune_model_id = get_finetune_model_id("t5-base-p-akksux-en-20220722-173018")

#model_max_length = 512
#batch_size = 8 if os.path.basename(base_model_id).startswith("t5-base") else 128

batch_size=1

num_train_epochs = 30

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

def readLangs(max_length_akk=5000, max_length_en=5000, length_threshold=100):
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
    # Split every line into pairs and normalize
    ###Translate from Akkadian cuneiform to English
    train_pairs_cuneiform_translate = [[normalizeString_akk(akk_train[l], use_prefix=True, task="Translate"), normalizeString_en(en_train[l], use_prefix=False)] for l in range(len(akk_train))]
    test_pairs_cuneiform_translate = [[normalizeString_akk(akk_test[l], use_prefix=True, task="Translate"), normalizeString_en(en_test[l], use_prefix=False)] for l in range(len(akk_test))]
    val_pairs_cuneiform_translate = [[normalizeString_akk(akk_val[l], use_prefix=True, task="Translate"), normalizeString_en(en_val[l], use_prefix=False)] for l in range(len(akk_val))]
    ###Translate from simple transliterated Akkadian to English
    train_pairs_simple_transliterated_translate = [[normalizeString_akk_transliterate_translate(transcription_train[l], use_prefix=True, task="Translate"), normalizeString_en(en_train[l], use_prefix=False)] for l in range(len(transcription_train))]
    test_pairs_simple_transliterated_translate = [[normalizeString_akk_transliterate_translate(transcription_test[l], use_prefix=True, task="Translate"), normalizeString_en(en_test[l], use_prefix=False)] for l in range(len(transcription_test))]
    val_pairs_simple_transliterated_translate = [[normalizeString_akk_transliterate_translate(transcription_val[l], use_prefix=True, task="Translate"), normalizeString_en(en_val[l], use_prefix=False)] for l in range(len(transcription_val))]
    ###Translate from grouped transliterated Akkadian to English
    train_pairs_group_transliterated_translate = [[normalizeString_akk_transliterate_minimal(transcription_train[l], use_prefix=True), normalizeString_en(en_train[l], use_prefix=False)] for l in range(len(transcription_train))]
    test_pairs_group_transliterated_translate = [[normalizeString_akk_transliterate_minimal(transcription_test[l], use_prefix=True), normalizeString_en(en_test[l], use_prefix=False)] for l in range(len(transcription_test))]
    val_pairs_group_transliterated_translate = [[normalizeString_akk_transliterate_minimal(transcription_val[l], use_prefix=True), normalizeString_en(en_val[l], use_prefix=False)] for l in range(len(transcription_val))]
    ###Transliterate from Akkadian Cuenfirom to Latin characters
    train_pairs_transliterate = [[normalizeString_akk(akk_train[l], use_prefix=True, task="Transliterate"), normalizeString_akk_transliterate(transcription_train[l], use_prefix=False)] for l in range(len(akk_train))]
    test_pairs_transliterate = [[normalizeString_akk(akk_test[l], use_prefix=True, task="Transliterate"), normalizeString_akk_transliterate(transcription_test[l], use_prefix=False)] for l in range(len(akk_test))]
    val_pairs_transliterated = [[normalizeString_akk(akk_val[l], use_prefix=True, task="Transliterate"), normalizeString_akk_transliterate(transcription_val[l], use_prefix=False)] for l in range(len(akk_val))]
    ###Group transliterated Akkadian into words
    train_pairs_transliterate_group = [[normalizeString_akk_transliterate_translate(transcription_train[l], use_prefix=True, task="Group"), normalizeString_akk_transliterate_minimal(transcription_train[l], use_prefix=False)] for l in range(len(akk_train))]
    test_pairs_transliterate_group = [[normalizeString_akk_transliterate_translate(transcription_test[l], use_prefix=True, task="Group"), normalizeString_akk_transliterate_minimal(transcription_test[l], use_prefix=False)] for l in range(len(akk_test))]
    val_pairs_transliterate_group = [[normalizeString_akk_transliterate_translate(transcription_val[l], use_prefix=True, task="Group"), normalizeString_akk_transliterate_minimal(transcription_val[l], use_prefix=False)] for l in range(len(akk_val))]
    ###Translate from English to cuneiform Akkadian
    rev_train_pairs_cuneiform_translate = [[normalizeString_en(en_train[l], use_prefix=True, task="Translate", target="cuneiform"), normalizeString_akk(akk_train[l], use_prefix=False)] for l in range(len(akk_train))]
    rev_test_pairs_cuneiform_translate = [[normalizeString_en(en_test[l], use_prefix=True, task="Translate", target="cuneiform"), normalizeString_akk(akk_test[l], use_prefix=False)] for l in range(len(akk_test))]
    rev_val_pairs_cuneiform_translate = [[normalizeString_en(en_val[l], use_prefix=True, task="Translate", target="cuneiform"), normalizeString_akk(akk_val[l], use_prefix=False), ] for l in range(len(akk_val))]
    ###Translate from English to simple transliterated Akkadian
    rev_train_pairs_simple_transliterated_translate = [[normalizeString_en(en_train[l], use_prefix=True, task="Translate", target="transliteration", characters="simple"), normalizeString_akk_transliterate(transcription_train[l], use_prefix=False)] for l in range(len(en_train))]
    rev_test_pairs_simple_transliterated_translate = [[normalizeString_en(en_test[l], use_prefix=True, task="Translate", target="transliteration", characters="simple"), normalizeString_akk_transliterate(transcription_test[l], use_prefix=False)] for l in range(len(en_test))]
    rev_val_pairs_simple_transliterated_translate = [[normalizeString_en(en_val[l], use_prefix=True, task="Translate", target="transliteration", characters="simple"), normalizeString_akk_transliterate(transcription_val[l], use_prefix=False)] for l in range(len(en_val))]
    ###Translate from English to grouped transliterated Akkadian
    rev_train_pairs_group_transliterated_translate = [[normalizeString_en(en_train[l], use_prefix=True, task="Translate", target="transliteration", characters="group"), normalizeString_akk_transliterate_minimal(transcription_train[l], use_prefix=False)] for l in range(len(en_train))]
    rev_test_pairs_group_transliterated_translate = [[normalizeString_en(en_test[l], use_prefix=True, task="Translate", target="transliteration", characters="group"), normalizeString_akk_transliterate_minimal(transcription_test[l], use_prefix=False)] for l in range(len(en_test))]
    rev_val_pairs_group_transliterated_translate = [[normalizeString_en(en_val[l], use_prefix=True, task="Translate", target="transliteration", characters="group"), normalizeString_akk_transliterate_minimal(transcription_val[l], use_prefix=False)] for l in range(len(en_val))]
    ###Convert from transliterated Akkadian to cuneiform
    rev_train_pairs_transliterate = [[normalizeString_akk_rev_transliterate(transcription_train[l], use_prefix=True), normalizeString_akk(akk_train[l], use_prefix=False)] for l in range(len(transcription_train))]
    rev_test_pairs_transliterate = [[normalizeString_akk_rev_transliterate(transcription_test[l], use_prefix=True), normalizeString_akk(akk_test[l], use_prefix=False)] for l in range(len(transcription_test))]
    rev_val_pairs_transliterate = [[normalizeString_akk_rev_transliterate(transcription_val[l], use_prefix=True), normalizeString_akk(akk_val[l], use_prefix=False)] for l in range(len(transcription_val))]
    ###Merge all data sets
    train_pairs = train_pairs_cuneiform_translate + train_pairs_simple_transliterated_translate + train_pairs_group_transliterated_translate + train_pairs_transliterate + train_pairs_transliterate_group + rev_train_pairs_cuneiform_translate + rev_train_pairs_simple_transliterated_translate + rev_train_pairs_group_transliterated_translate + rev_train_pairs_transliterate
    test_pairs = test_pairs_cuneiform_translate + test_pairs_simple_transliterated_translate + test_pairs_group_transliterated_translate + test_pairs_transliterate + test_pairs_transliterate_group + rev_test_pairs_cuneiform_translate + rev_test_pairs_simple_transliterated_translate + rev_test_pairs_group_transliterated_translate + rev_test_pairs_transliterate
    val_pairs = val_pairs_cuneiform_translate + val_pairs_simple_transliterated_translate + val_pairs_group_transliterated_translate + val_pairs_transliterated  + val_pairs_transliterate_group + rev_val_pairs_cuneiform_translate + rev_val_pairs_simple_transliterated_translate + rev_val_pairs_group_transliterated_translate + rev_val_pairs_transliterate
    pairs = train_pairs + test_pairs + val_pairs
    print("Examples:")
    print(f"", {train_pairs_cuneiform_translate[1][0]}, " -> ", {train_pairs_cuneiform_translate[1][1]})
    print(f"", {train_pairs_transliterate[1][0]}, " -> ", {train_pairs_transliterate[1][1]})
    print(f"", {train_pairs_transliterate_group[1][0]}, " -> ", {train_pairs_transliterate_group[1][1]})
    print(f"", {train_pairs_group_transliterated_translate[1][0]}, " -> ", {train_pairs_group_transliterated_translate[1][1]})
    print(f"Total pairs imported: {len(pairs)}")
    train_pairs = trim_pairs(train_pairs, max_length_akk, max_length_en, length_threshold)
    val_pairs = trim_pairs(val_pairs, max_length_akk, max_length_en, length_threshold)
    test_pairs = trim_pairs(test_pairs, max_length_akk, max_length_en, length_threshold)
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0].split()))
    max_length_pair_1 = max(pairs, key=lambda pair: len(pair[1].split()))
    print("Largest number of words in pair[0]:")
    print(f"Word Count: {len(max_length_pair_0[0].split())}, Content: {max_length_pair_0[0]}")
    print("Largest number of words in pair[1]:")
    print(f"Word Count: {len(max_length_pair_1[1].split())}, Content: {max_length_pair_1[1]}")
    mean_length_pair_0 = sum(len(pair[0].split()) for pair in pairs) / len(pairs)
    mean_length_pair_1 = sum(len(pair[1].split()) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in Akkadian: {mean_length_pair_0:.2f}")
    print(f"Mean number of tokens in English: {mean_length_pair_1:.2f}")
    return train_pairs, val_pairs, test_pairs, pairs

# Read your data
max_length = 28
cuneiform_pad = max_length
cuneiform_to_english_pad = int(max_length*0.62)
transliterated_to_english_pad = int(max_length*0.71)
train_pairs, val_pairs, test_pairs, pairs = readLangs(5000, 5000, max_length)

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



# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(finetune_model_id if is_finetune else base_model_id)
#model_path = "<model_path>"
#tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_path)



# Extract unique characters for new tokens
def extract_unique_characters(dataset, column):
    unique_chars = set()
    for example in dataset:
        unique_chars.update(set(example[column]))
    return unique_chars

# Extract unique tokens from English dataset
def extract_unique_tokens(dataset, column):
    unique_tokens = set()
    for example in dataset:
        tokens = example[column].split()  # Adjust token splitting if necessary
        unique_tokens.update(tokens)
    return unique_tokens

train_unique_tokens_akk = extract_unique_tokens(translations['train'], 'akk')
val_unique_tokens_akk = extract_unique_tokens(translations['val'], 'akk')
test_unique_tokens_akk = extract_unique_tokens(translations['test'], 'akk')

unique_tokens_akk = train_unique_tokens_akk.union(val_unique_tokens_akk).union(test_unique_tokens_akk)


train_unique_tokens_en = extract_unique_tokens(translations['train'], 'en')
val_unique_tokens_en = extract_unique_tokens(translations['val'], 'en')
test_unique_tokens_en = extract_unique_tokens(translations['test'], 'en')

unique_tokens_en = train_unique_tokens_en.union(val_unique_tokens_en).union(test_unique_tokens_en)

# Get current tokenizer vocabulary
current_vocab = set(tokenizer.get_vocab().keys())

# Find new tokens that are not in the current vocabulary
new_tokens_akk = unique_tokens_akk - current_vocab
new_tokens_en = unique_tokens_en - current_vocab

# Create a single set that includes all unique tokens from both sets, excluding duplicates
unique_new_tokens = new_tokens_akk.symmetric_difference(new_tokens_en)

# Add new tokens to the tokenizer
if unique_new_tokens:
    tokenizer.add_tokens(list(new_tokens_akk))

#if new_tokens_en:
    #tokenizer.add_tokens(list(new_tokens_en))

# Resize model token embeddings to accommodate new tokens
model.resize_token_embeddings(len(tokenizer))

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
metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    # print(preds)
    preds, labels = eval_preds
    #print('preds:',preds[0])
    # print('len:',preds[0].shape)
    if isinstance(preds, tuple):
        preds = preds[0]
    print('preds:',preds)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # if data_args.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

print("saving every ", int(len(train_pairs)/batch_size))
# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"../results/{model_id}",
    evaluation_strategy="steps",
    learning_rate=2*2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    fp16=False,
    save_steps=25000,
    eval_steps=25000,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    #predict_with_generate = True,
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
)

random.seed(9679)
print(random.choice(train_pairs))
print(random.choice(val_pairs))
print(random.choice(test_pairs))

# Train the model
trainer.train()
trainer.push_to_hub("akk-110m")
