"""
Author: Rohit Kundu
Date: 10 Oct, 2021
Goal: Create labeled data for disfluency detection (sequence tagging) where Input: Disfluent sentences and Output: Labels (at word-level) (0 -> fluent word, 1 -> disfluent word)

Imp: word comparison starts from the end of the sentence so that the latest uttered words are tagged as fluent and previously uttered words are tagged disfluent (in case of repetition for example).

Usage: python prepare-labeled-dataset.py [-h] [--disfluent DISFLUENT] [--fluent FLUENT] [--output-dir OUTPUT_DIR]
                                  [--generated-disfluent GENERATED_DISFLUENT] [--generated-labels GENERATED_LABELS]

Example: python prepare-labeled-dataset.py --disfluent data/parallel_data/synthetic-bengali.dis --fluent data/parallel_data/synthetic-bengali.flu --output-dir data/labeled_data/ --generated-disfluent synthetic-bengali.dis --generated-labels synthetic-bengali.labels

"""
import os
import argparse
from os import path

parser = argparse.ArgumentParser(description='Create labeled data for disfluency detection')
parser.add_argument('--disfluent', '-d', default='data/parallel_data/data.dis', type=str, help='Disfluent File')
parser.add_argument('--fluent', '-f', default='data/parallel_data/data.flu', type=str, help='Fluent File')
parser.add_argument('--output-dir', '-o', default='data/labeled_data/', type=str, help='Path of Generated Files')
parser.add_argument('--generated-disfluent', '-gd', default='data.dis', type=str, help='Name of the Generated Disfluent File')
parser.add_argument('--generated-labels', '-gl', default='data.labels', type=str, help='Name of the Generated labels File')
args = parser.parse_args()


if not path.exists(args.disfluent) or not path.exists(args.fluent):
    parser.error(f"Invalid file `{args.disfluent}` or `{args.fluent}`")

os.system(f"mkdir -p {args.output_dir}")

# Is subsequence at the word level (not at the character level)?
def isSubSequence(str1, str2):
    m = len(str1)
    n = len(str2)
 
    j = 0   
    i = 0    
 
    while j < m and i < n:
        if str1[j] == str2[i]:
            j = j+1

        i = i + 1
 
    return j == m

with open(args.disfluent, 'r') as dis, open(args.fluent, 'r') as flu, open(f"{args.output_dir}/{args.generated_disfluent}", 'w') as dis_out, open(f"{args.output_dir}/{args.generated_labels}", 'w') as label_out:
    dis = dis.readlines()
    flu = flu.readlines()
    subsequence = 0

    for dis_line, flu_line in zip(dis, flu):
        
        dis_line = dis_line.strip()
        flu_line = flu_line.strip()

        if flu_line == "None":
            flu_line = ""

        dis_words = dis_line.split()
        flu_words = flu_line.split()

        # Find those sentence pairs where fluent sentence is a subsequence of disfluent sentence
        if isSubSequence(flu_words, dis_words): 
            
            subsequence += 1

            i = len(dis_words) - 1
            j = len(flu_words) - 1
            
            labels = [1] * len(dis_words) # 0 means this word is part of disfluency

            while i >= 0:
                if j >= 0 and dis_words[i] == flu_words[j]:
                    labels[i] = 0 # Means dis_words[i] is not disfluent
                    j -= 1
            
                i -= 1

            # Sanity Check: fluent sentence does not contain extra words at the beginning
            if j != -1:
                subsequence -= 1
                continue
            
            dis_out.write(dis_line + "\n")
            label_out.write(' '.join(map(str, labels)) + "\n")

    print("{} fluent sentences are subsequence of corresponding disfluent sentence out of {} sentences".format(subsequence, len(dis)))
