from datasets import load_dataset, load_from_disk
import random
import os
from itertools import chain

preprocessing_num_workers = 32

def get_random_percentage_subset(dataset, percentage, seed=0):
    random.seed(seed)
    total_samples = len(dataset)
    num_samples = int(total_samples * (percentage / 100))
    indices = list(range(total_samples))
    random.shuffle(indices)
    subset_indices = indices[:num_samples]
    return dataset.select(subset_indices)


def tokenize_and_save_datasets(tokenizer, save_dir, seed=0):
    train_dataset = load_dataset("./datasets/SlimPajama-627B", split='train')
    val_dataset = load_dataset("./datasets/SlimPajama-627B", split='validation')

    train_dataset = get_random_percentage_subset(train_dataset, 100, seed)
    val_dataset = get_random_percentage_subset(val_dataset, 100, seed)

    train_column = list(train_dataset.features)
    val_column = list(val_dataset.features)

    def tokenize_function(data):
        output = tokenizer(data["text"])
        return output

    print("tokenizer processing may take time...")
    tokenized_train = train_dataset.shuffle().map(
        tokenize_function, 
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=train_column,
        )
    tokenized_val = val_dataset.shuffle().map(
        tokenize_function, 
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=val_column,
        )
    print("completes!")
    print("*" * 30)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tokenized_train.save_to_disk(os.path.join(save_dir, "tokenized_train"))
    tokenized_val.save_to_disk(os.path.join(save_dir, "tokenized_val"))
    print(f"tokenized datasets saved to {save_dir}")


def load_tokenized_datasets(save_dir):
    tokenized_train = load_from_disk(os.path.join(save_dir, "tokenized_train"))
    tokenized_val = load_from_disk(os.path.join(save_dir, "tokenized_val"))
    return tokenized_train, tokenized_val


def group_and_save_datasets(train_dataset, val_dataset, save_dir, max_length):
    def group_texts(data):
        # Concatenate all texts.
        concatenated_dataset = {
            k: list(chain(*data[k])) for k in data.keys()}
        total_length = len(concatenated_dataset[list(data.keys())[0]])
        # We drop the small remainder, and if the total_length < max_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_length]
                for i in range(0, total_length, max_length)]
            for k, t in concatenated_dataset.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    grouped_train = train_dataset.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
    )
    grouped_val = val_dataset.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    grouped_train.save_to_disk(os.path.join(save_dir, "grouped_train"))
    grouped_val.save_to_disk(os.path.join(save_dir, "grouped_val"))
    print(f"grouped datasets saved to {save_dir}")

    return train_dataset, val_dataset


def load_grouped_datasets(save_dir):
    train_dataset = load_from_disk(os.path.join(save_dir, "grouped_train"))
    val_dataset = load_from_disk(os.path.join(save_dir, "grouped_val"))
    return train_dataset, val_dataset


def traindata_loaders(tokenizer, max_length=1024, seed=0, grouped_dir="./datasets/grouped_slimpajama", tokenized_dir="./datasets/tokenized_slimpajama"):
    grouped_dir = grouped_dir + "_" + str(max_length)
    if os.path.exists(os.path.join(grouped_dir, "grouped_train")) and os.path.exists(os.path.join(grouped_dir, "grouped_val")):
        print("#### Grouped datasets already exist. Loading from disk...")
        train_dataset, val_dataset = load_grouped_datasets(grouped_dir)
    else:
        if os.path.exists(os.path.join(tokenized_dir, "tokenized_train")) and os.path.exists(os.path.join(tokenized_dir, "tokenized_val")):
            print("#### Tokenized datasets already exist. Loading from disk...")
            tokenized_train, tokenized_val = load_tokenized_datasets(tokenized_dir)
        else:
            print("#### Tokenized datasets do not exist. Tokenizing and saving datasets...")
            tokenize_and_save_datasets(tokenizer, tokenized_dir, seed)
            tokenized_train, tokenized_val = load_tokenized_datasets(tokenized_dir)

        train_dataset, val_dataset = group_and_save_datasets(tokenized_train, tokenized_val, grouped_dir, max_length=max_length)
        
    return train_dataset, val_dataset