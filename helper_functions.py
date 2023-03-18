import os
import re

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_metric
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


def load_exist_dataset_from_tsv(data_dir, train_path, test_path):
    train_df = pd.read_csv(os.path.join(data_dir, train_path), sep="\t")
    test_df = pd.read_csv(os.path.join(data_dir, test_path), sep="\t")
    return train_df, test_df


def select_only_english(df):
    return df[df["language"] == "en"]


def remove_unnecessary_columns(df):
    return df.drop(["language", "source", "test_case"], axis=1)


def split_train_valid(df):
    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)
    return train_df, valid_df


def create_huggingface_dataset(train_df, valid_df, test_df):
    train_dataset = Dataset.from_pandas(train_df, split="train", preserve_index=False)
    valid_dataset = Dataset.from_pandas(valid_df, split="validation", preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, split="test", preserve_index=False)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset, "valid": valid_dataset})
    return dataset


def save_dataset_to_hub(dataset, repo_name):
    # Note that this was done on google colab, so you need to login using huggingface-cli and tokens
    dataset.push_to_hub(repo_name)


def load_dataset_from_tsv_and_save_to_hub(data_dir, train_path, test_path, repo_name):
    train_df, test_df = load_exist_dataset_from_tsv(data_dir, train_path, test_path)
    train_df = select_only_english(train_df)
    test_df = select_only_english(test_df)
    train_df = remove_unnecessary_columns(train_df)
    test_df = remove_unnecessary_columns(test_df)
    train_df, valid_df = split_train_valid(train_df)
    dataset = create_huggingface_dataset(train_df, valid_df, test_df)
    save_dataset_to_hub(dataset, repo_name)


# Preprocess text (username and link placeholders)
def remove_hashtags_links_mentions(text: str) -> str:
    """Remove hashtags links and mentions from text

    Args:
        text (str): the text to be processed

    Returns:
        str: the processed text
    """
    text = re.sub(r"http\S+", "", text)
    text = text.replace(".", ". ").replace("?", "? ").replace("!", "! ")
    new_text = []
    # TODO: emoji replacement

    for t in text.split(" "):
        t = "" if t.startswith("@") and len(t) > 1 else t
        t = "" if t.startswith("#") and len(t) > 1 else t
        new_text.append(t)
    return " ".join(new_text)


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="micro")["f1"]
    return {"accuracy": accuracy, "f1": f1}


def define_mlp_model():
    return Pipeline(
        [
            ("vec", TfidfVectorizer(analyzer="word"))(
                "r5", MLPClassifier(random_state=1, hidden_layer_sizes=(500, 250, 500, 250, 100), max_iter=500)
            )
        ]
    )


def train_mlp(dataset):
    dataset = load_dataset("nouman-10/exist-en")

    train_dataset = dataset["train"]
    valid_dataset = dataset["valid"]
    test_dataset = dataset["test"]

    train_features = [row["text"] for row in train_dataset]
    valid_features = [row["text"] for row in valid_dataset]
    test_features = [row["text"] for row in test_dataset]

    train_labels_binary = [row["task1"] for row in train_dataset]
    valid_labels_binary = [row["task1"] for row in valid_dataset]
    test_labels_binary = [row["task1"] for row in test_dataset]

    train_labels_fine_grained = [row["task2"] for row in train_dataset]
    valid_labels_fine_grained = [row["task2"] for row in valid_dataset]
    test_labels_fine_grained = [row["task2"] for row in test_dataset]

    print("Binary MLP Model Classification")
    binary_model = define_mlp_model()

    binary_model.fit(train_features, train_labels_binary)

    print("Results on Validation Set")
    binary_preds = binary_model.predict(valid_features)
    print(classification_report(valid_labels_binary, binary_preds))

    print("Results on Test Set")
    binary_preds = binary_model.predict(test_features)
    print(classification_report(test_labels_binary, binary_preds))

    print("Fine Grained MLP Model Classification")
    fine_grained_model = define_mlp_model()

    fine_grained_model.fit(train_features, train_labels_fine_grained)

    print("Results on Validation Set")
    fine_grained_preds = fine_grained_model.predict(valid_features)
    print(classification_report(valid_labels_fine_grained, fine_grained_preds))

    print("Results on Test Set")
    fine_grained_preds = fine_grained_model.predict(test_features)
    print(classification_report(test_labels_fine_grained, fine_grained_preds))
