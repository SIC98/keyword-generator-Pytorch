from collections import defaultdict, Counter
from datasets import load_dataset
from utils import preprocess

dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')
dataset = dataset.map(preprocess)


def find_duplicate_ngrams(text, n):
    words = text.split()
    ngrams = defaultdict(int)

    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams[ngram] += 1

    duplicates = [ngram for ngram, count in ngrams.items() if count > 1]

    return duplicates


duplicates = [len(find_duplicate_ngrams(text, 3))
              for text in dataset['train']['Prompt']]

print(Counter(duplicates))
