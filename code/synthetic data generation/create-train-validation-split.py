"""
Author  : Rohit Kundu
Date    : 10 May, 2022
Goal    : Create train & validation set for finetuning MuRIL model on synthetic disfluency data
Usage   : python create-train-validation-split.py [-h] [--data-dir DATA_DIR]
                                        [--max-disfluency MAX_DISFLUENCY]
                                        [--max-training MAX_TRAINING]
                                        [--max-validation MAX_VALIDATION]
Example : python create-train-validation-split.py --data-dir noisy/hindi
"""

import os
import argparse

parser = argparse.ArgumentParser(description='Create train and validation splits')
parser.add_argument('--data-dir', '-d', default='noisy/bengali/', type=str, help='Data Directory which contains 1-disfluencies.dis, 2-disfluencies.flu, etc. files')
parser.add_argument('--max-disfluency', '-m', default=5, type=int, help=f"Maximum number of disfluencies in a sentence")
parser.add_argument('--max-training', '-t', default=8000, type=int, help=f"Maximum number of sentences in the training set")
parser.add_argument('--max-validation', '-v', default=500, type=int, help=f"Maximum number of sentences in the validation set")
args = parser.parse_args()

DATA_DIR = args.data_dir
MAX_NUM_OF_DISFLUENCY = args.max_disfluency
NUM_TRAINING_SENTENCES = args.max_training
NUM_VALIDATION_SENTENCES = args.max_validation

for typ in ['flu', 'dis']:
    os.system(f"rm -f  {DATA_DIR}/train.{typ} {DATA_DIR}/valid.{typ}")

for i in range(1, MAX_NUM_OF_DISFLUENCY + 1):
    for typ in ['flu', 'dis']:
        os.system(f"head -n {NUM_TRAINING_SENTENCES} {DATA_DIR}/{i}-disfluencies.{typ} >> {DATA_DIR}/train.{typ}")
        os.system(f"tail -n {NUM_VALIDATION_SENTENCES} {DATA_DIR}/{i}-disfluencies.{typ} >> {DATA_DIR}/valid.{typ}")

os.system(f"sed -i 's/{{//g' {DATA_DIR}/train.dis")
os.system(f"sed -i 's/}}//g' {DATA_DIR}/train.dis")
os.system(f"sed -i 's/{{//g' {DATA_DIR}/valid.dis")
os.system(f"sed -i 's/}}//g' {DATA_DIR}/valid.dis")
