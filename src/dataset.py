import copy
import json
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding



class WildGuard(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        def encode(examples):
            return tokenizer(
                examples["prompt"],
                examples["response"],
                truncation=True,
                padding="max_length",
            )

        dataset = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
        dataset = dataset.map(
            lambda example: {
                "labels": int(example["response_harm_label"] == "harmful")
            },
            batched=False,
        )
        self.dataset = dataset.map(encode, batched=True)
        if "roberta" in tokenizer.name_or_path:
            self.dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask","labels"],
        )
        else:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class ToxicChat(Dataset):
    def __init__(self, tokenizer):
        super().__init__()

        def encode(examples):
            return tokenizer(
                examples["user_input"], truncation=True, padding="max_length"
            )

        dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        dataset = dataset.map(
            lambda example: {"labels": example["toxicity"]}, batched=True
        )

        self.dataset = dataset.map(encode, batched=True)
        if "roberta" in tokenizer.name_or_path:
            self.dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask","labels"],
        )
        else:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class HarmBench(Dataset):
    def __init__(self, tokenizer):
        super().__init__()

        # Load the custom JSON data
        with open("data/harmbench_eval.json", "r") as file:
            data = json.load(file)

        label_to_id = {"unsafe": 1, "safe": 0}
        # Convert data to dataset format
        self.dataset = [
            {
                "prompt": item["prompt"],
                "response": item["response"],
                "labels": label_to_id[item["label"]],
            }
            for item in data
        ]

        # Encode dataset using tokenizer
        def encode(example):
            return tokenizer(
                example["prompt"],
                example["response"],
                truncation=True,
                padding="max_length",
            )

        # Tokenize the data
        self.dataset = [
            encode(example) | {"labels": example["labels"]} for example in self.dataset
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        # Convert to tensors
        item = {key: torch.tensor(val) for key, val in item.items()}
        return item


class OpenaiModeration(Dataset):
    def __init__(self, tokenizer):
        super().__init__()

        # Load the custom JSON data
        with open("data/openai_moderation_eval.json", "r") as file:
            data = json.load(file)

        # Convert data to dataset format
        self.dataset = [
            {"prompt": item["prompt"], "labels": item["toxicity"]} for item in data
        ]

        # Encode dataset using tokenizer
        def encode(example):
            return tokenizer(example["prompt"], truncation=True, padding="max_length")

        # Tokenize the data
        self.dataset = [
            encode(example) | {"labels": example["labels"]} for example in self.dataset
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        # Convert to tensors
        item = {key: torch.tensor(val) for key, val in item.items()}
        return item


class SFTDataset(Dataset):
    def __init__(self, tokenizer, prompt_file, split="train") -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = []

        with open(prompt_file, "r") as f:
            prompts = json.load(f)
        self.prompts = [x["instruction"].strip() for x in prompts]
        random.seed(42)
        random.shuffle(self.prompts)
        num_vals = int(len(self.prompts) * 0.1)

        if split == "train":
            self.prompts = self.prompts[num_vals:]
        elif split == "val":
            self.prompts = self.prompts[:num_vals]

        print(len(self.prompts))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt = self.prompts[index]
        item = self.encode(prompt)

        return item

    def encode(self, prompt):
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_ids.insert(0, self.tokenizer.bos_token_id)
        prompt_ids.append(self.tokenizer.eos_token_id)

        labels = copy.deepcopy(prompt_ids)
        mask = [1] * len(prompt_ids)

        return {"input_ids": prompt_ids, "labels": labels, "attention_mask": mask}


class KDDataset(Dataset):
    def __init__(self, tokenizer, json_file, split) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        assert self.tokenizer.padding_side == "left"
        with open(json_file) as f:
            data = json.load(f)[1:]

        self.chats = []
        self.labels = [1 if x["unsafe-score"] > 0.5 else 0 for x in data]
        for x in data:
            if "response" not in x:
                self.chats.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": x["prompt"]}], tokenize=False
                    )
                )
            else:
                self.chats.append(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": x["prompt"]},
                            {"role": "assistant", "content": x["response"]},
                        ],
                        tokenize=False,
                    )
                )
        assert len(self.chats) == len(self.labels)

        random.seed(42)
        random.shuffle(self.chats)
        num_vals = int(len(self.chats) * 0.1)

        if split == "train":
            self.chats = self.chats[num_vals:]
            self.labels = self.labels[num_vals:]
        elif split == "val":
            self.chats = self.chats[:num_vals]
            self.labels = self.labels[:num_vals]

    def __len__(self):
        return len(self.chats)

    def __getitem__(self, index):
        chat = self.chats[index]
        label = self.labels[index]
        return chat, label


class BERTDataset(Dataset):
    def __init__(self, tokenizer, json_file) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        # assert self.tokenizer.padding_side == "left"
        with open(json_file) as f:
            data = json.load(f)

        self.prompts = []
        self.responses = []
        self.labels = [1 if x["unsafe-score"] > 0.5 else 0 for x in data]
        if "soft_label" in data[0]:
            self.soft_labels = [x["soft_label"] for x in data]
        else:
            self.soft_labels = None
        for x in data:
            self.prompts.append(x["prompt"])
            if "response" not in x:
                self.responses.append("")
            else:
                self.responses.append(x["response"])
        assert len(self.prompts) == len(self.labels)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt = self.prompts[index]
        response = self.responses[index]
        label = self.labels[index]
        inputs = self.tokenizer(
            prompt, response, padding=False, add_special_tokens=True, truncation=True
        )
        inputs["labels"] = label
        if self.soft_labels is not None:
            inputs["soft_labels"] = self.soft_labels[index]
        
        return inputs


def get_dataloader(stage, tokenizer, input_file, batch_size=16, shuffle=True):
    if stage == "sft":
        tr_dataset = SFTDataset(tokenizer, input_file, split="train")
        val_dataset = SFTDataset(tokenizer, input_file, split="val")

        tr_dataloader = DataLoader(
            tr_dataset,
            batch_size,
            shuffle=shuffle,
            collate_fn=DataCollatorForSeq2Seq(tokenizer),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size,
            shuffle=False,
            collate_fn=DataCollatorForSeq2Seq(tokenizer),
        )

    elif stage == "kd":
        dataset = BERTDataset(tokenizer, input_file)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        X = np.arange(len(dataset))
        y = np.array(dataset.labels)
        train_idx, val_idx = next(sss.split(X, y))

        tr_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        tr_labels = [y[idx] for idx in train_idx]
        target = torch.tensor(tr_labels)
        class_sample_count = torch.tensor(
            [(target == t).sum() for t in torch.unique(target, sorted=True)]
        )
        weight = 1.0 / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in target])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        tr_dataloader = DataLoader(
            tr_dataset,
            batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer),
            sampler=sampler,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer),
        )

    elif stage == "toxic-chat":
        dataset = ToxicChat(tokenizer)
        tr_dataloader = None
        val_dataloader = DataLoader(dataset, batch_size, shuffle=False)

    elif stage == "harmbench":
        dataset = HarmBench(tokenizer)
        tr_dataloader = None
        val_dataloader = DataLoader(dataset, batch_size, shuffle=False)

    elif stage == "openai-moderation":
        dataset = OpenaiModeration(tokenizer)
        tr_dataloader = None
        val_dataloader = DataLoader(dataset, batch_size, shuffle=False)

    elif stage == "wildguard":
        dataset = WildGuard(tokenizer)
        tr_dataloader = None
        val_dataloader = DataLoader(dataset, batch_size, shuffle=False)

    else:
        raise NotImplementedError

    return tr_dataloader, val_dataloader


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for data_name in [
        "wildguard", 
        "openai-moderation", 
        "harmbench",
        "toxic-chat",
    ]:
        _, dataloader = get_dataloader(data_name, tokenizer, None, shuffle=False)
        label_num_ones = 0
        total = 0
        for batch in dataloader:
            label_num_ones += batch["labels"].sum().item()
            total += batch["labels"].size(0)
        print(f"{data_name=}, {label_num_ones=}, {total=}")
