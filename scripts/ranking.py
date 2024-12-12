from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

def get_shuffled_dataset(raw_datasets):
    raw_datasets = raw_datasets.shuffle(seed=42)
    raw_datasets["train"] = raw_datasets["train"].select(range(len(raw_datasets["train"]) // 2))
    return raw_datasets.flatten_indices()


def get_length_sorted(raw_datasets):
    # Add a new column with the length of 'text_chosen'
    raw_datasets = raw_datasets.map(
        lambda x: {"text_length": len(x["chosen"][-1]["content"])},
        batched=False,
        num_proc=8
    )
    
    # Sort the dataset by the new 'text_length' column in descending order
    raw_datasets = raw_datasets.sort("text_length", reverse=True)
    
    # Slice the "train" split to keep the first half
    raw_datasets["train"] = raw_datasets["train"].select(range(len(raw_datasets["train"]) // 2))
    
    # Remove the added 'text_length' column if not needed anymore
    raw_datasets = raw_datasets.remove_columns("text_length")
    
    return raw_datasets

# def custom_tokenizer(text):
#     # Split on non-alphanumeric or non-comma boundaries
#     tokens = re.split(r'[\W_]+', text)
#     # Remove empty strings and return tokens
#     return [token for token in tokens if token]

def get_complexity_sorted(raw_datasets):
    # Combine all text fields from the training set into a single collection
    train_texts = [
        entry["prompt"] + " " + entry["chosen"][-1]["content"] + " " + entry["rejected"][-1]["content"]
        for entry in raw_datasets["train"]
    ]

    # Fit TF-IDF vectorizer on the training data
    tfidf_vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_vectorizer.fit(train_texts)

    # Precompute useful values
    tfidf_vocab = tfidf_vectorizer.vocabulary_
    idf_values = tfidf_vectorizer.idf_
    lowest_tfidf_value = idf_values.min()

    # Function to compute complexity for a single example
    def compute_complexity_for_example(example):
        def average_tfidf(text):
            tokens = text.split()
            tfidf_scores = [
                idf_values[tfidf_vocab[token]]
                if token in tfidf_vocab
                else lowest_tfidf_value
                for token in tokens
            ]
            return np.mean(tfidf_scores) if tfidf_scores else lowest_tfidf_value

        return {
            "text_prompt_complexity": average_tfidf(example["prompt"]),
            "text_chosen_complexity": average_tfidf(example["chosen"][-1]["content"]),
            "text_rejected_complexity": average_tfidf(example["rejected"][-1]["content"]),
        }

    # Add complexity columns to the training dataset
    raw_datasets["train"] = raw_datasets["train"].map(
        compute_complexity_for_example,
        num_proc=8,
        desc="Computing complexity for train set"
    )

    # Sort training data by `text_prompt_complexity` (or another complexity field)
    raw_datasets["train"] = raw_datasets["train"].sort("text_prompt_complexity").select(range(len(raw_datasets["train"]) // 2))

    return raw_datasets
