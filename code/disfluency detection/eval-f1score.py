"""
Author: Rohit Kundu
Date: 25 December, 2021
Goal: Evaluate Precision, Recall, F1score of the predicted fluent sentences from a file containing disfluent sentences and a file containing reference labels
Usage: py eval-f1score.py [-h] --disfluent DISFLUENT --reference-labels REFERENCE_LABELS --latex-table-output
Example: py eval-f1score.py --disfluent data/labeled_data/my-swbd/test.dis --reference-labels data/labeled_data/my-swbd/test.labels --prediction predictions/5yTrrzcH/swbd-test.flu_prediction

Important: Precision, Recall, F1 score is calculated at word level labels
"""

import argparse
import os
from os import path
import sys

parser = argparse.ArgumentParser(description='Evaluate Precision, Recall, F1 score of the predicted fluent sentences')
parser.add_argument('--disfluent', '-d', required=True, type=str, help='Input file with disfluent sentences in each line')
parser.add_argument('--reference-labels', '-r', required=True, type=str, help='Reference Input file with token wise labels (0/1) in each line')
parser.add_argument('--prediction', '-p', default='test.flu_prediction', type=str, help='Predicted fluent sentences in each line')
parser.add_argument('--latex-table-output', '-l', action='store_true', help='Print latex code the row of the results table')

args = parser.parse_args()

if not path.exists(args.disfluent):
    parser.error(f"Invalid input file `{args.disfluent}`")

if not path.exists(args.reference_labels):
    parser.error(f"Invalid input file `{args.reference_labels}`")

if not path.exists(args.prediction):
    parser.error(f"Invalid input file `{args.prediction}`")

os.system(f"python prepare-labeled-dataset.py --disfluent {args.disfluent} --fluent {args.prediction} --output-dir . --generated-disfluent test.dis --generated-labels test.predicted.labels")

with open(args.reference_labels, 'r') as ref, open('test.predicted.labels', 'r') as pred:
    ref_lines = ref.readlines()
    pred_lines = pred.readlines()
    if len(ref_lines) != len(pred_lines):
        parser.error("Error: No of lines mismatch")
    else:

        tp, tn, fp, fn = 0, 0, 0, 0
        for ref_line, pred_line in zip(ref_lines, pred_lines):
            ref_labels = ref_line.split()
            pred_labels = pred_line.split()

            if len(ref_labels) != len(pred_labels):
                parser.error("Error: Data mismatch")

            for l1, l2 in zip(ref_labels, pred_labels):
                if l1 == l2 and l1 == '1':
                    tp += 1
                elif l1 == l2 and l1 == '0':
                    tn += 1
                elif l1 == '1':
                    fn += 1
                else:
                    fp += 1
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        
        print("Precision: {:0.2f}".format(precision * 100))
        print("Recall: {:0.2f}".format(recall * 100))
        print("Accuracy: {:0.2f}".format(100 * (tp + tn) / (tp + tn + fp + fn)))
        print("F1 Score: {:0.2f}".format(100 * f1))

        # precision, recall, f1 score
        if args.latex_table_output:
            print("precision & recall & f1 : {:0.2f} & {:0.2f} & {:0.2f}".format(precision * 100, recall * 100, 100 * f1))
        
        print("True label=0: {:0.2f}%".format(100 * (tn + fp) / (tp + tn + fp + fn)))
        print("True label=1: {:0.2f}%".format(100 * (tp + fn) / (tp + tn + fp + fn)))


os.remove('test.predicted.labels')
os.remove('test.dis')
