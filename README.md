## Getting Started
### Directory Structure

The following directories provides scripts for CTVAE

- `scripts`   Contains files for training and sampling sentences
- `evaluation_scripts` Contains files for evaluation of the sampled sentences
- `scripts/BERT.py` Run this code to generate sentence embeddings using BERT-base-uncased pretrained model
- `scripts/train_datasets.py`  This is the main file to train the model
- `scripts/sample_prior.py`  Run this code to sample sentences from prior
- `scripts/sample_post.py` Run this code to sample sentences from posterior
- `scripts/sample_interpolation.py` Run this code for fine tuned sample generation

### Prerequisites
- `pytorch 1.4.0`
- `pytorch-pretrained-bert 0.6.2`
- `torchtext 0.6.0`
- `spacy 3.1.0`
- `python -m spacy download en_core_web_sm`

### Dataset
Download the yelp dataset from [here](https://drive.google.com/drive/folders/1P5zUZ1XNq4642hSc8sU_Yy-veMInC-Ca?usp=sharing).
Please download other opensource datasets and cite them appropriately. 


It has following files

- `.data/${name}/${name}/train/data.txt`
- `.data/${name}/${name}/train/labels.txt`
- `.data/${name}/${name}/val/data.txt`
- `.data/${name}/${name}/val/labels.txt`
- `.data/${name}/${name}/test/data.txt`
- `.data/${name}/${name}/test/labels.txt`

### BERT embedding files
Run the following to create BERT embedding files required for the training process.
  ```bash
    mkdir ./BERT
    python scripts/BERT.py .data/${name}/${name}/train/data.txt ./BERT/${name}_train.pt
    python scripts/BERT.py .data/${name}/${name}/val/data.txt ./BERT/${name}_val.pt
    python scripts/BERT.py .data/${name}/${name}/test/data.txt ./BERT/${name}_test.pt
  ```

## Training the model
### Prameters
You can specify the parameters according to your design choices in `scripts/train_datasets.py`

### Command
python scripts/train_datasets.py --with_bert --epochs 20 --data <name> --save_dir <output dir> --z_dim 256 --z_dim_1 256
###  Output
The directory `saved_${name}` contains the checkpoints:
- `vae.bin` stores the checkpoint state

## Sampling sentences from prior

### Command
`python sample_prior.py --with_bert --save_dir saved_${name} --output_dir prior_output_${name} --z_dim 256 --z_dim_1 256`

###  Output
The directory `prior_output_${name}` will store the following:
- `pos_prior.txt` stores the sentences which are positive
- `neg_prior.txt` stores the sentences which are negtaive
- This code will output `max_senti` and `min_senti` for future use

## Sampling sentences from posterior (Text-style-transfer)

### Command
- Please provide the values of `min_senti` and `max_senti` as achieved from the training data
`python sample_post.py --with_bert --save_dir saved_${name} --output_dir post_output_${name} --z_dim 256 --z_dim_1 256`

###  Output
The directroy `post_output_${name}` will store the following:
- `pos_post.txt` stores the sentences which are positive
- `neg_post.txt` stores the sentences which are negtaive
- `original.txt` stores the original sentences whose posterior was taken into account
- `labels.txt` stores the label of the original sentences


## Sampling sentences using interpolation (Fine-Tuning attribute)

### Command
- Please provide the values of `min_senti` and `max_senti` as achieved from the training data
`python sample_interpolation.py --with_bert --save_dir saved_${name} --output_dir interpolated_output_${name} --z_dim 256 --z_dim_1 256`

###  Output
The directory `interpolated_output_${name}` will store the following:
- `post.i.txt` stores the sentences for `i-th` sentiment level
- `original.txt` stores the original sentences whose posterior was taken into account
- `labels.txt` stores the label of the original sentences

# Evaluation

## Accuracy
We will train a SVM classifier with BERT sentence encodings as input. To get the sentence encodings, we need to run a background service (`bert-serving-server`) that will provide the sentence encodings.

- The BERT pretrained model can be downloaded from [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip).
- Following two packages are required for the BERT
  ```bash
    pip install bert-serving-server
    pip install bert-serving-client
  ```
- Start the BERT sentence encoding server as follows

  ```bash
    bert_port=8190
    bert-serving-start -model_dir bert_base_uncased -num_worker=4 -port=${bert_port} -max_seq_len=NONE
  ```

### Train a BERT based SVM classifier
Train a sklearn SVM classifier with BERT sentence encodings as input.

`mkdir bert_svm`

`python evaluation_scripts/train_bert_classifier.py .data/${name}/${name}/train .data/${name}/${name}/val bert_svm/${name} ${bert_port}`

### Classify using the SVM classifier
Classify using the trained sklean SVM classifier which accepts BERT sentence encodings as input.

`# python evaluation_scripts/bert_classify.py bert_svm/${name} <input_file> <output_file> [port]`

- for prior
  ```bash
    for f in pos_prior neg_prior
    do
      python evaluation_scripts/bert_classify.py bert_svm/${name} prior_output_${name}/${f}.txt prior_output_${name}/${f}.bert_labels ${bert_port}
    done
  ```
- for posterior
  ```bash
    for f in pos_post neg_post
    do
      python evaluation_scripts/bert_classify.py bert_svm/${name} post_output_${name}/${f}.txt post_output_${name}/${f}.bert_labels ${bert_port}
    done
  ```

### Get accuracy score
`label_ratio.py` reports the ratio of each label in a (labels) file.
- for prior
  ```bash
    for f in pos_prior.bert_labels neg_prior.bert_labels
    do
      echo $f
      echo evaluation_scripts/label_ratio.py prior_output_${name}/$f
    done
  ```
- for posterior
  ```bash
    for f in pos_post.bert_labels neg_post.bert_labels
    do
      echo $f
      echo evaluation_scripts/label_ratio.py post_output_${name}/$f
    done
  ```

## On interpolated sentiment output
### Get sentiment score using Stanford CoreNLP

Get the Stanford CoreNLP software

  ```bash
    wget http://nlp.stanford.edu/software/stanford-corenlp-4.0.0.zip
    unzip stanford-corenlp-4.0.0.zip
    cd stanford-corenlp-4.0.0
    wget http://nlp.stanford.edu/software/stanford-corenlp-4.0.0-models-english.jar
    # RUN
    stan_port=9100
    java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 5000 -port ${stan_port}
  ```

Run `stan_score.py` to get the sentiment scores using Stanford CoreNLP

  `# python evaluation_scripts/stan_score.py [port(optional)] <input_file> <output_file>`

  ```bash
    for i in {0..34}
    do
      python evaluation_scripts/stan_score.py ${stan_port} interpolated_output_${name}/post.$i.txt interpolated_output_${name}/post.$i.score
    done
  ```

### Overall sentiment score

Get the average score corresponding to:
- Negative input, Negative output
- Negative input, Positive output
- Positive input, Negative output
- Positive input, Positive output

  `# python evaluation_scripts/stan_score_summary.py <scores_file> <original_labels>`

  ```bash
    for i in {0..34}
    do
      python evaluation_scripts/stan_score_summary.py interpolated_output_${name}/post.$i.score interpolated_output_${name}/labels.txt > interpolated_output_${name}/post.$i.score_sum
    done

    ## Prints scores in columns, starting from 0th to 34th interpolated values.
    paste interpolated_output_${name}/post.${0..34}.score_sum | tail -n6
  ```

### Jaccard Score

Get the jaccard score for interpolated sentences corresponding to:
- Negative input, Negative output
- Negative input, Positive output
- Positive input, Negative output
- Positive input, Positive output

`num_per_sample` is 100 by default, i.e., 100 samples per sentence (as set during the generation of interpolated sentiment sentences).

  `# python evaluation_scripts/jaccard_score.py <original_text> <original_labels> <ouput_text> <stanford_scores> [num_per_sample (optional)]`

  ```bash
    for i in {0..34}
    do
      python evaluation_scripts/jaccard_score.py interpolated_output_${name}/original.txt interpolated_output_${name}/labels.txt interpolated_output_${name}/post.$i.txt interpolated_output_${name}/post.$i.score > interpolated_output_${name}/post.$i.jaccard
    done

    ## Prints scores in columns, starting from 0th to 34th interpolated values.
    paste interpolated_output_${name}/post.${0..34}.jaccard | tail -n6
  ```

## On interpolated output
### Get Probablity Scores From SVM

- run the "BERT serving" server

  `bert-serving-start -model_dir bert-base-uncased -num_worker=2 -port=8190 -max_seq_len=64`


- run the following

  ```bash
  ./evaluation_scripts/prob_scores.sh <directory> <model_file> <samples_per_sentence>
  ./evaluation_scripts/prob_scores.sh output_music music.svm 10
  ```

- to show the scores, run following

  ```bash
  paste <directory>/prob/post.{0..19}.analysis
  paste output_music/prob/post.{0..19}.analysis
  ```

### OVERLAP SCORE

- run the following

  ```bash
  ./evaluation_scripts/overlap_scores.sh <directory> <samples_per_sentence>
  ./evaluation_scripts/overlap_scores.sh output_music 10
  ```

- to show the scores, run following

  ```bash
  paste <directory>/prob/post.{0..19}.overlap_scores
  paste output_music/prob/post.{0..19}.overlap_scores
  ```

### Probability Score after Cosine and Overlap Threshold

- Run the above two steps before running this,
  - probability scores generation script, `./evaluation_scripts/prob_scores.sh`
  - overlap scores script, `./evaluation_scripts/overlap_scores.sh`

- run the "BERT serving" server

  `bert-serving-start -model_dir bert-base-uncased -num_worker=2 -port=8190 -max_seq_len=64`


- run the following

  ```bash
  ./evaluation_scripts/prob_scores_with_thresh.sh <directory> <cosine_threshold> <overlap_threshold>
  ./evaluation_scripts/prob_scores_with_thresh.sh output_music 0.7 0.2
  ```

- to show the scores, run following

  ```bash
  paste <directory>/prob/post.{0..19}.cosine_analysis
  paste output_music/prob/post.{0..19}.cosine_analysis
  ```

### Merge 20 interpolated files into 10 interpolated files

- run the following, to create a directory `merged` inside the input directory

  ```bash
  ./evaluation_scripts/merge.sh <directory>
  ```
