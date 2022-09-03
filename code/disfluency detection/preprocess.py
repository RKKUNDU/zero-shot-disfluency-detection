import argparse
from os import path
import os

"""
Author: Rohit Kundu
Date  : 26 December, 2021
Goal  : Preprocess text files containing sentences in Indian languages
        -> normalization
        -> tokenization
        -> remove punctuations
        -> trim spaces
        -> detokenization

conda activate indic
py preprocess.py --input data/parallel_data/500-bengali-test-sentences/test.dis --output data/parallel_data/500-bengali-test-sentences/test.processed.dis --lang bn

py preprocess.py --input data/parallel_data/500-bengali-test-sentences/test.flu --output data/parallel_data/500-bengali-test-sentences/test.processed.flu --lang bn
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lang', '-l', required=True, type=str, help='Language e.g. bn, mr')
parser.add_argument('--input', '-i', required=True, type=str, help='Input file')
parser.add_argument('--output', '-o', required=True, type=str, help='Output file')
args = parser.parse_args()

if not path.exists(args.input):
    parser.error(f"Invalid file `{args.input}`")


LANG = args.lang
SRC = args.input
NORMALIZED = f"{SRC}.nor"
TOKENIZED = f"{NORMALIZED}.tok"
PUNC = f"{TOKENIZED}.punc"
TRIMMED = f"{PUNC}.trim"

# Normalization
os.system(f"python /home/development/rohitk/tools/indic_nlp_library/indicnlp/normalize/indic_normalize.py {SRC} {NORMALIZED} {LANG}")

# Tokenization
os.system(f"python /home/development/rohitk/tools/indic_nlp_library/indicnlp/tokenize/indic_tokenize.py {NORMALIZED} {TOKENIZED} {LANG}")

# REMOVE Punctuations
os.system(f"""sed -e 's/[।.!?,\"]\+//g' {TOKENIZED} > {PUNC}""")

# Trim spaces
os.system("awk '{$1=$1;print}' " + "{} > {}".format(PUNC, TRIMMED))

os.system(f"python /home/development/rohitk/tools/indic_nlp_library/indicnlp/tokenize/indic_detokenize.py {TRIMMED} {args.output} {LANG}")

os.system(f"rm {NORMALIZED} {TOKENIZED} {PUNC} {TRIMMED}")