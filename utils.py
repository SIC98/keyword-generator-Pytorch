import numpy as np
import datasets
import nltk
import re

nltk.download("punkt", quiet=True)

metric = datasets.load_metric("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result['gen_len'] = np.mean(prediction_lens)

    result = {k: round(v, 4) for k, v in result.items()}
    return result


def batch_tokenize_preprocess(batch, tokenizer):
    source = batch['Prompt']
    source_tokenized = tokenizer(
        source, padding='max_length', truncation=False, max_length=1024
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch['labels'] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in source_tokenized['input_ids']
    ]
    return batch


def preprocess(data):
    text = data['Prompt']

    # Remove spaces between consecutive single digit numbers
    text = re.sub(r'(\b\d\b) +(\b\d\b)', r'\1\2', text)
    # Remove spaces between numbers and alphabets (e.g. '4 k' -> '4k')
    text = re.sub(r'(\b\d+\b) (\b[a-zA-Z]\b)', r'\1\2', text)
    # Remove spaces before and after of "-"
    text = re.sub(r' *- *', '-', text)
    # Replace '_' in to ' '
    text = text.replace('_', ' ')
    # Reduce two or more spaces to one space
    text = re.sub(r' +', ' ', text)
    # Remove spaces in front of "," & Only one space after ","
    text = re.sub(r' *, *', ', ', text)

    data['Prompt'] = text
    return data
