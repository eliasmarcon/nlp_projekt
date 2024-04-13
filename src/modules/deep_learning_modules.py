import pandas as pd
import torch
import numpy as np
import datetime

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split

def read_data(file_path = '../datasets/01_preprocessed_datasets/dataset_preprocessed_no_transformation.csv'):
    
    # load the dataset
    df = pd.read_csv(file_path)

    # map hate to 1 and non-hate to 0
    df['label'] = df['label'].map({'hate': 1, 'not_hate': 0})

    return df['text'].values, df['label'].values


def tokenize_data(sentences, labels, tokenizer, max_length):

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                           # Sentence to encode.
                            add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                            max_length = max_length,           # Pad & truncate all sentences.
                            padding = 'max_length',         # padding to max_len
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',          # Return pytorch tensors.
                            truncation=True                 # truncate the sentence to max_len
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    return input_ids, attention_masks, labels


def create_data_split(input_ids, attention_masks, labels, seed_val):
    
    generator1 = torch.Generator().manual_seed(seed_val)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2], generator=generator1)
    
    return train_dataset, val_dataset, test_dataset


def create_data_loader(dataset, batch_size, random = True):
    
    dataloader = DataLoader(
            dataset,  # The training samples.
            sampler = RandomSampler(dataset) if random else SequentialSampler(dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
    
    return dataloader

def flat_accuracy(preds, labels):
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_loss_logits(modeltype, model, b_input_ids, b_input_mask, b_labels):
    
    if modeltype == "distilbert":
        
        loss = model(b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels).loss
        
        logits = model(b_input_ids,
                       attention_mask=b_input_mask,
                       labels=b_labels).logits
        
        
    elif modeltype in ["bert_base", "albert_base", "transformer", "roberta"]:
    
        loss = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask,
                    labels=b_labels).loss
        
        logits = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels).logits
        
    return loss, logits