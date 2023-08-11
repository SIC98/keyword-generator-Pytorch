from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_lightning import seed_everything
from dataclasses import dataclass, field
from datasets import load_from_disk
import evaluate
import argparse

from utils import get_data_until_kth_comma, inference_preprocess, batch_inference

seed_everything(42)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--input_type', type=str,
                        choices=['one_keyword', 'two_keyword'])

    args = parser.parse_args()

    if args.input_type == 'one_keyword':
        split_index = 1
    elif args.input_type == 'two_keyword':
        split_index = 2

    tokenizer = GPT2Tokenizer.from_pretrained(
        args.model_name_or_path, padding_side='left'
    )

    # add the EOS token as PAD token to avoid warnings
    model = GPT2LMHeadModel.from_pretrained(
        args.model_name_or_path, pad_token_id=tokenizer.eos_token_id
    )

    model = model.to(args.device)
    if args.fp16:
        model = model.half()

    dataset = load_from_disk("lexica-data")
    del dataset['train']
    del dataset['validation']

    dataset = dataset.map(lambda data: inference_preprocess(data, "Prompt"))
    dataset = dataset.filter(lambda example: example["total_comma"] > 2)
    dataset = dataset.map(
        lambda batch: batch_inference(
            batch, model, args.input_type, tokenizer, args.device),
        batched=True,
        batch_size=args.batch_size,
    )

    predictions = []
    references = []

    for data in dataset['test']:
        prediction = get_data_until_kth_comma(
            data['generated_text'],
            data['total_comma'] + 1,
            False
        )
        reference = data['Prompt']

        prediction = prediction.split(', ', split_index)[1]
        reference = reference.split(', ', split_index)[1]

        predictions.append(prediction)
        references.append(reference)

    rouge_metric = evaluate.load('rouge')

    rouge_score = rouge_metric.compute(
        predictions=predictions, references=references, use_stemmer=True
    )

    print(rouge_score)


if __name__ == '__main__':
    main()
