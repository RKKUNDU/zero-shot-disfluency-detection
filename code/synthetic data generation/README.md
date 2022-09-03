#### Input and Outut Data
* `data/` contains cleaned fluent sentences in Bengali, Hindi, Marathi from PMIndia corpus.
* `noisy/` contains the synthetically generated disfluent sentences

#### Disfluency Injection Code
* `multiple-bengali-disfluency-injection.py` is used to generate synthetic disfluency correction data in Bengali.
* `multiple-hindi-disfluency-injection.py` is used to generate synthetic disfluency correction data in Hindi.
* `multiple-marathi-disfluency-injection.py` is used to generate synthetic disfluency correction data in Marathi.
* `multiple-marathi-disfluency-injection-only-subset-of-disfluency-types.py` is used to generate synthetic disfluency correction data in Marathi. Here, we use a subset of disfluency types.

#### Others
* `preprocess.py` is used to preprocess the generated data before used for training.
* `create-train-validation-split.py` is used to create train, validation sets for training.
