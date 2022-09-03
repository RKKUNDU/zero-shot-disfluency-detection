"""
Author: Rohit Kundu
Date: 9 October, 2021
Goal: Generate fluent translations from a file containing disfluent sentences
Usage: CUDA_VISIBLE_DEVICES=3 python test-generate.py [-h] --file FILE --checkpoint CHECKPOINT [--out OUT] [--verbose]
"""

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
import numpy as np
from datasets import Dataset
import argparse
from os import path
import sys

parser = argparse.ArgumentParser(description='Generate fluent translations from a file containing disfluent sentences')
parser.add_argument('--file', '-f', required=True, type=str, help='Input file with disfluent sentences in each line')
parser.add_argument('--checkpoint', '-c', required=True, type=str, help='Path to the checkpoint directory')
parser.add_argument('--out', '-o', type=str, help='Output file')
parser.add_argument('--verbose', '-v', action='store_true', help='If set, prints Input & Output sentences, otherwise just prints the fluent translations in each line')
args = parser.parse_args()

if args.file and path.exists(args.file):
    with open(args.file, 'r') as file:
        test_sentences = file.readlines()
        test_sentences = [sentence.strip() for sentence in test_sentences]
else:
    parser.error(f"Invalid input file `{args.file}`")

if not path.exists(f"{args.checkpoint}/pytorch_model.bin") or not path.exists(f"{args.checkpoint}/tokenizer.json") or not path.exists(f"{args.checkpoint}/config.json"):
    parser.error(f"Invalid checkpoint directory `{args.checkpoint}`")

if args.out:
    writer = open(args.out, 'w')
    sys.stdout = writer

model_checkpoint = args.checkpoint
label_list = ['is_fluent', 'is_disfluent'] # 0 -> is_fluent , 1 -> is_disfluent
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

data_collator = DataCollatorForTokenClassification(tokenizer)


train_args = TrainingArguments(
    'evaluation',
    # per_device_eval_batch_size=2,
    # per_device_train_batch_size=2,
)


trainer = Trainer(
    model,
    args=train_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

def test_tokenizer(examples):
    tokenized_inputs = tokenizer(examples["disfluent"], truncation=True, max_length=512, is_split_into_words=True)
    return tokenized_inputs

# Evaluate on blind sentences
test_dict = {
                'disfluent': [sentence.split() for sentence in test_sentences]
            }

test_dataset = Dataset.from_dict(test_dict)
test_dataset = test_dataset.map(test_tokenizer, batched=True)

predictions, _, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=2)
    
tokenized_for_word_ids = tokenizer(test_dataset["disfluent"], truncation=True, is_split_into_words=True)

for i in range(predictions.shape[0]):
    actual_input = test_dataset["disfluent"][i]
    word_ids = tokenized_for_word_ids.word_ids(i)[1:-1] # Remove [CLS] & [SEP] token's word_id
            
    if args.verbose:
        print("Input:\t   ", ' '.join(actual_input))
    # print(tokenized_input)
    # print(word_ids)
    # print(predictions[i][1:1 + len(tokenized_input)])
    
    previous_word_idx = None
    disfluent = 0 # Count of (predicted) disfluent subwords of a word
    fluent = 0 # count of (predicted) fluent subwords of a words
    fluent_sentence = []
    
    for idx, prediction in enumerate(predictions[i][1:1 + len(word_ids)]): # Remove [CLS] & [SEP] & PAD TOKEN predictions
            
        # We add/ignore the previous word (based on how many subwords of the word were predicted disfluent).
        # Added if count(fluent subwords) >= count(disfluent subwords)
        if word_ids[idx] != previous_word_idx:
            if previous_word_idx is not None and fluent >= disfluent:
                fluent_sentence.append(actual_input[previous_word_idx])
            
            fluent, disfluent = 0, 0
        
        if prediction == 0:
            fluent += 1
        else:
            disfluent += 1
            
        previous_word_idx = word_ids[idx]
        
    # Don't forget to add the last word
    if previous_word_idx is not None and fluent >= disfluent:
        fluent_sentence.append(actual_input[previous_word_idx])

    if args.verbose:
        print("Prediction:",' '.join(fluent_sentence))
        print()
    else:
        print(' '.join(fluent_sentence))

if args.out:
    writer.close()
