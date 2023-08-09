import re


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
