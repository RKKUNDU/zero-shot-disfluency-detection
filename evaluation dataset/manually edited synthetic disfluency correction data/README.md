This directory contains manually-edited synthetic disfluency correction data in Bengali, Hindi, Malayalam, and Marathi.

Each directory contains four files:
* `test.labels` containing tokenwise binary labels (0 &rarr; fluent, 1 &rarr; disfluent) for all the disfluent sentences
* `test.processed.dis` containing preprocessed disfluent sentences in each line
* `test.processed.flu` containing preprocessed fluent sentences in each line
* `typewise-flu-dis-pairs.tsv` containing 3 columns disfluency type, fluent and disfluent sentences. 
