"""
Author  : Rohit Kundu
Goal    : Generate synthetic Bengali sentences having 1, 2, 3, .. , n disfluencies 
Imp     : 
            -> Input file is `data/data.nor.clean.clean2.clean3.clean4.bn`
            -> Output folder is `noisy/bengali/`
            -> Set the config parameters appropriately (e.g. MAX_NUM_OF_DISFLUENCY, NUM_EXAMPLE, etc.)
"""

import os, random
from indicnlp.syllable import  syllabifier
from collections import defaultdict


INPUT = 'data/data.nor.clean.clean2.clean3.clean4.bn'
NOISY_DIR = 'noisy/bengali/'
SYNONYMS_FILE = 'synonyms/cleaned_synonyms.bn'
os.system(f'mkdir -p {NOISY_DIR}')

## Config ##
MIN_LENGTH_OF_LONG_WORD = 12
MAX_NUM_OF_DISFLUENCY = 5
NUM_EXAMPLE = 20000
lang = 'bn'
MAX_TRY = 100

disfluency_mapping = {
    1: 'filler',
    2: 'word repetition',
    3: 'phrase repetition',
    4: 'pronoun repetition',
    5: 'partial word',
    6: 'missing syllables',
    7: 'pronoun correction',
    8: 'synonym correction'
}

disfluency_to_code = {v: k for k, v in disfluency_mapping.items()}
## Config ##


with open(INPUT, 'r') as file:
    flu_lines = file.readlines()

long_words_for_missing_syllables = {}
cnt = 0
dp = {}

# This will take long time
for line in flu_lines:
    indices = []
    line = line.replace('।', " ").replace(',', ' ').replace('!', ' ').replace('.', ' ').replace('?', ' ')

    for idx, word in enumerate(line.split()):
        if len(word) >= MIN_LENGTH_OF_LONG_WORD:
            if word in dp:
                syllables = dp[word]
            else:
                syllables = syllabifier.orthographic_syllabify_improved(word, lang)
                dp[word] = syllables

            if len(syllables) >= 7:
                indices.append(idx)

    long_words_for_missing_syllables[line] = indices
    if (cnt + 1) % 1000 == 0:
        print(f"Steps: {cnt + 1}/{len(flu_lines)}")
    
    cnt += 1

print("Time consuming task complete!!")

# Get synonyms
SYNONYMS = dict()
with open(SYNONYMS_FILE, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        word, synonym = line.split(', ')
        SYNONYMS[word] = synonym

PRONOUN_GROUPS = [
    ['আমি', 'আমাকে', 'আমার', 'আমরা', 'আমাদেরকে', 'আমাদের'],
    ['তুই', 'তোকে', 'তোর', 'তোরা', 'তোদেরকে', 'তোদের'],
    ['তুমি', 'তোমাকে', 'তোমার', 'তোমরা', 'তোমাদেরকে', 'তোমাদের'],
    ['আপনি', 'আপনাকে', 'আপনার', 'আপনারা', 'আপনাদেরকে', 'আপনাদের'],
    ['এ', 'একে', 'এর', 'এরা', 'এদেরকে', 'এদের'],
    ['ও', 'ওকে', 'ওর', 'ওরা', 'ওদেরকে', 'ওদের'],
    ['সে', 'তাকে', 'তার', 'তারা', 'তাদেরকে', 'তাদের'],
    ['ইনি', 'এঁকে', 'এনাকে', 'এঁর', 'এনার', 'এঁরা', 'এঁদেরকে', 'এনাদেরকে', 'এঁদের', 'এনাদের'],
    ['উনি', 'ওঁকে', 'ওনাকে', 'ওঁর', 'ওনার', 'ওঁরা', 'ওঁদেরকে', 'ওনাদেরকে', 'ওঁদের', 'ওনাদের'],
    ['তিনি', 'তাঁকে', 'তাঁর', 'তাঁরা', 'তাঁদেরকে', 'তাঁদের'],
    ['এটা', 'ওটা', 'এইটা', 'ওইটা', 'এই', 'ওই']
]
PRONOUNS = [pronoun for grp in PRONOUN_GROUPS for pronoun in grp]


def get_wordidx_having_synonyms(words):
    indices = []
    for idx, word in enumerate(words):
        if word in SYNONYMS.keys():
            indices.append(idx)

    return indices


def get_pronounidx(words):
    indices = []
    for idx, word in enumerate(words):
        if word in PRONOUNS:
            indices.append(idx)

    return indices


def get_long_wordidx(words):
    indices = []
    for idx, word in enumerate(words):
        if len(word) >= MIN_LENGTH_OF_LONG_WORD:
            indices.append(idx)

    return indices    

    
def pick_random_sentence():
    global flu_lines
    idx = random.randint(0, len(flu_lines) - 1)
    return flu_lines[idx]


def get_synonym_correction(word):
    return SYNONYMS[word]


def get_pronoun_correction(word):
    FILLERS = [
        '',
        'না',
        'না না',
        'না মানে'
    ]
    # Repeat a pronoun from the same group (except the pronoun)
    for pronoun_group in PRONOUN_GROUPS:
        if word in pronoun_group:
            new_group = pronoun_group.copy()
            new_group.remove(word)
            improper_pronoun = random.choices(new_group)[0]

            idx = random.randint(0, len(FILLERS) - 1)
            filler = (FILLERS[idx])

            return improper_pronoun + " " + filler            


def get_missing_syllables(word):
    syllables = syllabifier.orthographic_syllabify_improved(word, lang)
    if len(syllables) >= 7:
        length = random.choices([1, 2, 3, 4, 5], [0.35, 0.30, 0.25, 0.05, 0.05])[0]
        idx = random.randint(1, len(syllables) - length - 1)
        return ''.join(syllables[:idx] + syllables[idx + length:])
    else:
        return -1


def get_partial_word(word):
    syllables = syllabifier.orthographic_syllabify_improved(word, lang)
    length = random.choices([i for i in range(1, len(syllables) + 1)],
                            [len(syllables) / i for i in range(1, len(syllables) + 1)])[0]
    return ''.join(syllables[:length])


def get_filler():
    FILLERS = [
        'আরে',
        'মানে',
        'এই মানে',
        'না মানে',
        'ইয়ে',
        'ইয়ে মানে',
        'আচ্ছা',
        'ইশশ',
        'যাঃ',
        'উফ',
        'উফফ',
        'ধুর',
        'ইসে',
        'অ্যা',
        'হুম',
        'আঃ',
        'উ', 
        'উম',
        'যে',
        'উঃ',
    ]
    idx = random.randint(0, len(FILLERS) - 1)
    return FILLERS[idx]


def get_disfluent_sentence(fluent, disfluency):
    disfluent = ""
    for idx, word in enumerate(fluent.split()):
        if idx in disfluency:
            disfluent +=  "{"  + disfluency[idx] + "} "
            # disfluent += disfluency[idx] + " "
        
        disfluent += word + " "
    
    return disfluent.strip()


def inject_disfluencies(NUM_OF_DISFLUENCY):

    global long_words_for_missing_syllables

    print(f"Thread-{NUM_OF_DISFLUENCY} started")
    cnt = 0
    disfluency_type_cnts = defaultdict(int)
    fluent_file = open(f"{NOISY_DIR}/{NUM_OF_DISFLUENCY}-disfluencies.flu", 'w')
    disfluent_file = open(f"{NOISY_DIR}/{NUM_OF_DISFLUENCY}-disfluencies.dis", 'w')
    disfluent_types_file = open(f"{NOISY_DIR}/{NUM_OF_DISFLUENCY}-disfluencies.types", 'w')

    while cnt < NUM_EXAMPLE:

        disfluency_types = [random.randint(1, 8) for _ in range(NUM_OF_DISFLUENCY)]
        temp = disfluency_types.copy()
        sentence = pick_random_sentence()
        sentence = sentence.replace('।', " ").replace(',', ' ').replace('!', ' ').replace('.', ' ').replace('?', ' ')
        words = sentence.split()
        synonyms_idx = get_wordidx_having_synonyms(words)
        pronouns_idx = get_pronounidx(words)
        longwords_idx = get_long_wordidx(words)
        used_idx = []
        disfluency = {}
        broken = False

        tried = 0
        code = disfluency_to_code['synonym correction']
        while code in disfluency_types:
            if len(synonyms_idx) == 0:
                broken = True            
                break

            idx = random.choice(synonyms_idx)
            if idx not in used_idx:
                disfluency[idx] = get_synonym_correction(words[idx])
                used_idx.append(idx)
                disfluency_types.remove(code)

            tried += 1
            if tried == MAX_TRY:
                broken = True
                break

        if broken:
            continue

        code = disfluency_to_code['pronoun correction']
        while code in disfluency_types:
            if len(pronouns_idx) == 0:
                broken = True            
                break

            idx = random.choice(pronouns_idx)
            if idx not in used_idx:
                disfluency[idx] = get_pronoun_correction(words[idx])
                used_idx.append(idx)
                disfluency_types.remove(code)

            tried += 1
            if tried == MAX_TRY:
                broken = True
                break


        if broken:
            continue

        code = disfluency_to_code['pronoun repetition']
        while code in disfluency_types:
            if len(pronouns_idx) == 0:
                broken = True            
                break

            idx = random.choice(pronouns_idx)
            if idx not in used_idx:
                disfluency[idx] = words[idx]
                used_idx.append(idx)
                disfluency_types.remove(code)

            tried += 1
            if tried == MAX_TRY:
                broken = True
                break

        if broken:
            continue

        code = disfluency_to_code['missing syllables']
        while code in disfluency_types:
            if len(long_words_for_missing_syllables[sentence]) == 0:
                broken = True            
                break

            idx = random.choice(long_words_for_missing_syllables[sentence])
            if idx not in used_idx:
                ret = get_missing_syllables(words[idx])
                if ret != -1:
                    disfluency[idx] = ret
                    used_idx.append(idx)
                    disfluency_types.remove(code)

            tried += 1
            if tried == MAX_TRY:
                broken = True
                break

        if broken:
            continue

        code = disfluency_to_code['partial word']
        while code in disfluency_types:
            if len(longwords_idx) == 0:
                broken = True            
                break

            idx = random.choice(longwords_idx)
            if idx not in used_idx:
                disfluency[idx] = get_partial_word(words[idx])
                used_idx.append(idx)
                disfluency_types.remove(code)

            tried += 1
            if tried == MAX_TRY:
                broken = True
                break

        if broken:
            continue

        code = disfluency_to_code['word repetition']
        while code in disfluency_types:
            idx = random.randint(0, len(words) - 1)
            if idx not in used_idx:
                disfluency[idx] = words[idx]
                used_idx.append(idx)
                disfluency_types.remove(code)
            
            tried += 1
            if tried == MAX_TRY:
                broken = True
                break

        if broken:
            continue

        code = disfluency_to_code['filler']
        while code in disfluency_types:
            idx = random.randint(0, len(words) - 1)
            if idx not in used_idx:
                disfluency[idx] = get_filler()
                used_idx.append(idx)
                disfluency_types.remove(code)
    
            tried += 1
            if tried == MAX_TRY:
                broken = True
                break
        
        if broken:
            continue

        code = disfluency_to_code['phrase repetition']
        while code in disfluency_types:
            
            phrase_length = random.choices([2, 3, 4, 5], weights=[0.4, 0.3, 0.2, 0.1])[0] 
            if len(words) <= phrase_length:
                broken = True
                break

            idx = random.randint(0, len(words) - phrase_length)
            if idx in used_idx:
                tried += 1
                if tried == MAX_TRY:
                    broken = True
                    break
                continue

            # check I have not injected any disfluency in the words whose index ranges from [idx, idx + phrase_len -1] 
            for i in range(idx, idx + phrase_length):
                if i in disfluency:
                    break
            else:
                disfluency[idx] = " ".join(words[idx:idx + phrase_length])
                for i in range(idx, idx + phrase_length):
                    used_idx.append(i)

                disfluency_types.remove(code)
    
            tried += 1
            if tried == MAX_TRY:
                broken = True
                break
        
        if broken:
            continue

        # print(get_disfluent_sentence(sentence, disfluency))
        cnt += 1
        for typ in temp:
            disfluency_type_cnts[typ] += 1

        fluent_file.write(sentence)
        disfluent_file.write(get_disfluent_sentence(sentence, disfluency) + "\n")
        disfluent_types_file.write(" ".join(map(str, temp)) + "\n")
        

        if cnt % 500 == 0:
            print(f"Thread-{NUM_OF_DISFLUENCY}: {cnt}/{NUM_EXAMPLE} steps")

    print(f"Disfluency Type Counts ({NUM_OF_DISFLUENCY} disfluencies per sentence):", disfluency_type_cnts)

for NUM_OF_DISFLUENCY in range(1, MAX_NUM_OF_DISFLUENCY + 1):
    inject_disfluencies(NUM_OF_DISFLUENCY)
