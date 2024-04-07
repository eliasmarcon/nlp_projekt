# nlp_projekt

## Python Version
- 3.11.4

## Virtual Environment

If a virtual environment is wanted please execute the following code

```sh

python -m venv <environment_name>

# change into the environment and activate it (here a git bash terminal was opened)
source <environment_name>\Scripts\activate

```

## Install all required Packages

If you use the virtual environment make source its activated in your console (<environment_name> before your path). After this install the requirements with the following command.

```sh
pip install -r requirements.txt
```

## Dataset

additional dataset
- https://github.com/Vicomtech/hate-speech-dataset
- https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain


## Max Lengths Models and Datasets

| Modeltype | Dataset | Max Length |
| --- | --- | --- |
| bert_base     | dataset_preprocessed_no_transformation | 443 |
| bert_base     | dataset_preprocessed_stopwords         | 293 |
| distilbert    | dataset_preprocessed_no_transformation | 443 |
| distilbert    | dataset_preprocessed_stopwords         | 293 |
| albert_base   | dataset_preprocessed_no_transformation | 439 |
| albert_base   | dataset_preprocessed_stopwords         | 287 |
| roberta       | dataset_preprocessed_no_transformation | 521 |
| roberta       | dataset_preprocessed_stopwords         | 371 |
| transformer   | dataset_preprocessed_no_transformation | 443 |
| transformer   | dataset_preprocessed_stopwords         | 293 |


## Models trained

| Modeltype | Dataset | Learning Rate | Encoding | Trained? |
| --- | --- | --- | --- | --- |
| bert_base     | dataset_preprocessed_no_transformation | 0.0001   | 256, 512 | , YES |
| bert_base     | dataset_preprocessed_no_transformation | 0.00002  | 256, 512 | , YES |
| bert_base     | dataset_preprocessed_no_transformation | 0.00005  | 256, 512 | , YES |
| bert_base     | dataset_preprocessed_stopwords         | 0.0001   | 256, 512 | , YES |
| bert_base     | dataset_preprocessed_stopwords         | 0.00002  | 256, 512 | , YES |
| bert_base     | dataset_preprocessed_stopwords         | 0.00005  | 256, 512 | , YES |
| distilbert    | dataset_preprocessed_no_transformation | 0.0001   | 256, 512 | , YES |
| distilbert    | dataset_preprocessed_no_transformation | 0.00002  | 256, 512 | , YES |
| distilbert    | dataset_preprocessed_no_transformation | 0.00005  | 256, 512 | , YES |
| distilbert    | dataset_preprocessed_stopwords         | 0.0001   | 256, 512 | , YES |
| distilbert    | dataset_preprocessed_stopwords         | 0.00002  | 256, 512 | , YES |
| distilbert    | dataset_preprocessed_stopwords         | 0.00005  | 256, 512 | , YES |
| albert_base   | dataset_preprocessed_no_transformation | 0.0001   | 256, 512 |  |
| albert_base   | dataset_preprocessed_no_transformation | 0.00002  | 256, 512 |  |
| albert_base   | dataset_preprocessed_no_transformation | 0.00005  | 256, 512 |  |
| albert_base   | dataset_preprocessed_stopwords         | 0.0001   | 256, 512 |  |
| albert_base   | dataset_preprocessed_stopwords         | 0.00002  | 256, 512 |  |
| albert_base   | dataset_preprocessed_stopwords         | 0.00005  | 256, 512 |  |
| roberta       | dataset_preprocessed_no_transformation | 0.0001   | 256, 512 | , YES |
| roberta       | dataset_preprocessed_no_transformation | 0.00002  | 256, 512 | , YES |
| roberta       | dataset_preprocessed_no_transformation | 0.00005  | 256, 512 | , YES |
| roberta       | dataset_preprocessed_stopwords         | 0.0001   | 256, 512 | , YES |
| roberta       | dataset_preprocessed_stopwords         | 0.00002  | 256, 512 | , YES |
| roberta       | dataset_preprocessed_stopwords         | 0.00005  | 256, 512 | , YES |
| transformer   | dataset_preprocessed_no_transformation | 0.0001   | 256, 512 | , YES |
| transformer   | dataset_preprocessed_no_transformation | 0.00002  | 256, 512 | , YES |
| transformer   | dataset_preprocessed_no_transformation | 0.00005  | 256, 512 | , YES |
| transformer   | dataset_preprocessed_stopwords         | 0.0001   | 256, 512 |  |
| transformer   | dataset_preprocessed_stopwords         | 0.00002  | 256, 512 |  |
| transformer   | dataset_preprocessed_stopwords         | 0.00005  | 256, 512 |  |

