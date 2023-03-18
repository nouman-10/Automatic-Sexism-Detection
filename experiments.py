import helper_functions as f
from fine_tune import train_model
from mlm import MLM

# HuggingFace Hub username, change this to your username if you want to push the models to your own HuggingFace Hub account
HF_USERNAME = "nouman-10"

# Dataset Paths for the HuggingFace Datasets Hub
EXIST_DATASET = f"{HF_USERNAME}/exist-en"
SEMEVAL_EXIST_DATASET = f"{HF_USERNAME}/exist-semeval-combined-sexist-en"
ALL_COMBINED_DATASET = f"{HF_USERNAME}/combined-sexist-eval-en"


# HuggingFace Model Hub Paths for the binary fine-tuned models without Masked Language Modeling
binary_fine_tuned_models_without_mlm = {
    "bert_exist": f"{HF_USERNAME}/fine-tune-bert-exist",
    "roberta-exist": f"{HF_USERNAME}/fine-tune-roberta-exist",
    "bert-all-combined": f"{HF_USERNAME}/fine-tune-bert-combined",
    "roberta-all-combined": f"{HF_USERNAME}/fine-tune-roberta-combined",
    "bert-semeval-exist": f"{HF_USERNAME}/fine-tune-bert-sem-exist",
    "roberta-semeval-exist": f"{HF_USERNAME}/fine-tune-roberta-sem-exist",
}

mlm_models = {
    "bert-exist": "unsupervised-exist",
    "roberta-exist": "unsupervised-exist-rb",
    "bert-all-combined": "unsupervised-combined",
    "roberta-all-combined": "unsupervised-combined-rb",
}

# HuggingFace Model Hub Paths for the binary fine-tuned models with Masked Language Modeling
binary_fine_tuned_models_with_mlm = {
    "bert-exist": f"{HF_USERNAME}/fine-tune-bert-exist-mlm",
    "roberta-exist": f"{HF_USERNAME}/fine-tune-roberta-exist-mlm",
    "bert-all-combined": f"{HF_USERNAME}/fine-tune-bert-combined-mlm",
    "roberta-all-combined": f"{HF_USERNAME}/fine-tune-roberta-combined-mlm",
}

# HuggingFace Model Hub Paths for the fine-tuned fine-grained classification models with Masked Language Modeling
fine_grained_fine_tuned_models_with_mlm = {
    "bert-exist": f"{HF_USERNAME}/fine-tune-bert-exist-fine-grained",
    "roberta-exist": f"{HF_USERNAME}/fine-tune-roberta-exist-fine-grained",
    "bert-all-combined": f"{HF_USERNAME}/fine-tune-bert-combined-fine-grained",
    "roberta-all-combined": f"{HF_USERNAME}/fine-tune-roberta-combined-fine-grained",
}


def get_dataset_model_tokenizer_ckpts(model_path, mlm=False):
    if "exist" in model_path:
        dataset_name = EXIST_DATASET
    elif "combined" in model_path:
        dataset_name = ALL_COMBINED_DATASET
    else:
        dataset_name = SEMEVAL_EXIST_DATASET

    if "bert" in model_path:
        if not mlm:
            model_checkpoint = "bert-base-uncased"
        else:
            model_checkpoint = "unsupervised-exist" if "exist" in model_path else "unsupervised-combined"
        tokenizer_checkpoint = "bert-base-uncased"
    else:
        if not mlm:
            model_checkpoint = "roberta-base"
        else:
            model_checkpoint = "unsupervised-exist-rb" if "exist" in model_path else "unsupervised-combined-rb"
        tokenizer_checkpoint = "roberta-base"

    return dataset_name, model_checkpoint, tokenizer_checkpoint


if __name__ == "main":
    # The following code assumes that you have loaded the dataset into your huggingface or using my datasets

    print("Baseline MLP Model Classification")
    f.train_mlp()

    # The following code doesn't train the models at the moment and uses the already trained models to evaluate the performance. Set train=True to train the models again.
    print("Binary Fine-Tuned Model Classification without Masked Language Modeling")
    for model_name, model_path in binary_fine_tuned_models_without_mlm.items():
        print(f"Model: {model_name}")
        num_labels = 2
        dataset_name, model_checkpoint, tokenizer_checkpoint = get_dataset_model_tokenizer_ckpts(model_path)

        train_model(
            dataset_name=dataset_name,
            output_model_name=model_path,
            model_checkpoint=model_checkpoint,
            tokenizer_checkpoint=tokenizer_checkpoint,
            num_labels=num_labels,
            train=False,
        )

    # Trains the MLM models (Already trained available at the huggingface hub)
    print("Training MLM Models")
    for model_name, model_path in mlm_models.items():
        print(f"Model: {model_name}")
        dataset_name, output_model_name, model_checkpoint = get_dataset_model_tokenizer_ckpts(model_path, mlm=True)
        mlm = MLM(dataset_name=dataset_name, output_model_name=output_model_name, model_checkpoint=model_checkpoint,)
        mlm.train_and_push_to_hub()

    print("Binary Fine-Tuned Model Classification with Masked Language Modeling")
    for model_name, model_path in binary_fine_tuned_models_with_mlm.items():
        print(f"Model: {model_name}")
        num_labels = 2
        dataset_name, model_checkpoint, tokenizer_checkpoint = get_dataset_model_tokenizer_ckpts(model_path, mlm=True)

        train_model(
            dataset_name=dataset_name,
            output_model_name=model_path,
            model_checkpoint=model_checkpoint,
            tokenizer_checkpoint=tokenizer_checkpoint,
            num_labels=num_labels,
            train=False,
        )

    print("Fine-grained Fine-Tuned Model Classification with Masked Language Modeling")
    for model_name, model_path in fine_grained_fine_tuned_models_with_mlm.items():
        print(f"Model: {model_name}")
        num_labels = 2
        dataset_name, model_checkpoint, tokenizer_checkpoint = get_dataset_model_tokenizer_ckpts(model_path, mlm=True)

        train_model(
            dataset_name=dataset_name,
            output_model_name=model_path,
            model_checkpoint=model_checkpoint,
            tokenizer_checkpoint=tokenizer_checkpoint,
            num_labels=num_labels,
        )
