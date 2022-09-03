"""
Author: Rohit Kundu
Date: 9 October, 2021
Goal: Interactively generate fluent translations from the input disfluent sentences
Usage: CUDA_VISIBLE_DEVICES=3 python test-interactive.py [-h] --checkpoint CHECKPOINT [--verbose]
"""

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
from datasets import Dataset
import argparse
from os import path

parser = argparse.ArgumentParser(description='Interactively generate fluent translations from the input disfluent sentences')
parser.add_argument('--checkpoint', '-c', required=True, type=str, help='Path to the checkpoint directory')
parser.add_argument('--verbose', '-v', action="store_true", help='Verbose')
args = parser.parse_args()

if not path.exists(f"{args.checkpoint}/pytorch_model.bin") or not path.exists(f"{args.checkpoint}/tokenizer.json") or not path.exists(f"{args.checkpoint}/config.json"):
    parser.error(f"Invalid checkpoint directory `{args.checkpoint}`")

model_checkpoint = args.checkpoint
label_list = ['is_fluent', 'is_disfluent'] # 0 -> is_fluent , 1 -> is_disfluent
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

def test_tokenizer(examples):
    tokenized_inputs = tokenizer(examples["disfluent"], truncation=True, is_split_into_words=True)
    return tokenized_inputs

# Evaluate on blind input sentences
input_sentence = ""
while True:
    input_sentence = input("Enter a disfluent sentence: ").strip()
    if input_sentence.lower() == "q":
        break

    test_dict = {
                    'disfluent': [input_sentence.split()]
                }

    test_dataset = Dataset.from_dict(test_dict)
    test_dataset = test_dataset.map(test_tokenizer, batched=True)
    test_dataset = test_dataset.remove_columns(['disfluent'])

    prediction, _, _ = trainer.predict(test_dataset)
    prediction = np.argmax(prediction, axis=2)

    actual_input = input_sentence.split()
    tokenized_input = tokenizer.tokenize(input_sentence.split(), is_split_into_words=True)
    word_ids = tokenizer(input_sentence.split(), is_split_into_words=True).word_ids()[1:-1] # Remove [CLS] & [SEP] token's word_id
    
    if args.verbose:
        print("Tokenized Input:", tokenized_input)
        print("Predicted Labels:", prediction[0][1:1 + len(tokenized_input)])
        for subword, label in zip(tokenized_input, prediction[0][1:1 + len(tokenized_input)]):
            print(subword, label) 


    previous_word_idx = None
    disfluent = 0 # Count of (predicted) disfluent subwords of a word
    fluent = 0 # count of (predicted) fluent subwords of a words
    fluent_sentence = []
    
    for idx, predicted_label in enumerate(prediction[0][1:1 + len(tokenized_input)]): # Remove [CLS] & [SEP] & PAD TOKEN predictions
            
        # We add/ignore the previous word (based on how many subwords of the word were predicted disfluent).
        # Added if count(fluent subwords) >= count(disfluent subwords)
        if word_ids[idx] != previous_word_idx:
            if previous_word_idx is not None and fluent >= disfluent:
                fluent_sentence.append(actual_input[previous_word_idx])
            
            fluent, disfluent = 0, 0
        
        if predicted_label == 0:
            fluent += 1
        else:
            disfluent += 1
            
        previous_word_idx = word_ids[idx]
        
    # Don't forget to add the last word
    if previous_word_idx is not None and fluent >= disfluent:
        fluent_sentence.append(actual_input[previous_word_idx])

    print("Prediction:",' '.join(fluent_sentence))
    print()
