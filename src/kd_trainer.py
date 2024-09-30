import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

from src.dataset import get_dataloader
from src.utils import get_decay_parameter_names


class KDTrainer(object):
    def __init__(self, args) -> None:
        self.args = args

        wandb.init(
            reinit=True,
            config=args.as_dict(),
            project=args.wandb_project,
            name=args.exp_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.kd_model, padding_side="right"
        )
        self.tokenizer.model_max_length = args.max_length

        config = AutoConfig.from_pretrained(args.kd_model, num_labels=2)
        self.device = torch.cuda.current_device()

        if "deberta" in args.kd_model:
            config.max_position_embeddings = args.max_length
            self.model = AutoModelForSequenceClassification.from_pretrained(
                args.kd_model,
                config=config,
            )
        else:
            attn_implementation = "sdpa" if "bert-" in args.kd_model else None 
            self.model = AutoModelForSequenceClassification.from_pretrained(
                args.kd_model, config=config, attn_implementation=attn_implementation
            )
            # extend positional embedding by 2
            self.model.config.max_position_embeddings = args.max_length
            if "bert-" in args.kd_model:
                self.model.bert.embeddings.position_ids = torch.arange(
                    args.max_length
                ).expand((1, -1))
                self.model.bert.embeddings.token_type_ids = torch.zeros(
                args.max_length).expand((1, -1))
                orig_pos_emb = self.model.bert.embeddings.position_embeddings.weight
                self.model.bert.embeddings.position_embeddings.weight = torch.nn.Parameter(
                    torch.cat([orig_pos_emb, orig_pos_emb]))
            
            elif "roberta" in args.kd_model:
                self.model.roberta.embeddings.position_ids = torch.arange(
                    args.max_length
                ).expand((1, -1))

                self.model.roberta.embeddings.token_type_ids = torch.zeros(
                args.max_length).expand((1, -1))
                orig_pos_emb = self.model.roberta.embeddings.position_embeddings.weight
                self.model.roberta.embeddings.position_embeddings.weight = torch.nn.Parameter(
                    torch.cat([orig_pos_emb, orig_pos_emb]))
        
        self.model = self.model.to(self.device)
        self.tr_loader, self.val_loader = get_dataloader(
            "kd", self.tokenizer, args.kd_file, args.batch_size, shuffle=True
        )

        _, self.test_loader_toxic_chat = get_dataloader(
            "toxic-chat", self.tokenizer, None, args.batch_size, shuffle=False
        )

        _, self.test_loader_harmbench = get_dataloader(
            "harmbench", self.tokenizer, None, args.batch_size, shuffle=False
        )

        _, self.test_loader_openai = get_dataloader(
            "openai-moderation", self.tokenizer, None, args.batch_size, shuffle=False
        )

        _, self.test_loader_wildguard = get_dataloader(
            "wildguard", self.tokenizer, None, args.batch_size, shuffle=False
        )

        decay_parameters = get_decay_parameter_names(self.model)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

        t_total = len(self.tr_loader) * self.args.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, args.num_warmup_steps, t_total
        )

    def train(self):
        global_step = 1
        for epoch in tqdm(range(self.args.epochs), dynamic_ncols=True):

            t = tqdm(
                self.tr_loader,
                total=len(self.tr_loader),
                leave=False,
                dynamic_ncols=True,
            )
            for batch in t:
                batch_loss = []
                chunks = {
                    k: torch.chunk(v, self.args.grad_acc_steps, dim=0)
                    for k, v in batch.items()
                }
                num_chunks = len(chunks["input_ids"])

                self.optimizer.zero_grad()
                for i in tqdm(
                    range(num_chunks),
                    desc="gradient step",
                    dynamic_ncols=True,
                    leave=False,
                ):
                    self.model.train()
                    mini_batch = {k: v[i].to(self.device) for k, v in chunks.items()}
                    try:
                        soft_labels = mini_batch.pop("soft_labels")
                    except KeyError:
                        soft_labels = None
                    outputs = self.model(**mini_batch)
                    loss = outputs.loss
                    
                    if soft_labels is not None:
                        targets = torch.stack([1-soft_labels, soft_labels], dim=1)
                        logits = outputs.logits
                        log_probs = F.log_softmax(logits, 1)
                        kl_loss = F.kl_div(log_probs, targets, reduction="batchmean")
                        loss = 0.5 * (loss + kl_loss)
                    
                    loss = loss / self.args.grad_acc_steps

                    loss.backward()
                    batch_loss.append(loss.item())

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)

                self.optimizer.step()
                self.scheduler.step()

                wandb.log({"loss/train": sum(batch_loss)}, step=global_step)
                num_unsafe = (batch.labels).sum().item()
                t.set_description(
                    f"epoch: {epoch}, step {global_step}: {sum(batch_loss): .4f}, num unsafe: {num_unsafe}"
                )
                global_step += 1

            metrics = self.eval(self.val_loader, split="val")
            wandb.log(metrics, step=global_step)

        test_metrics = self.eval(
            self.test_loader_toxic_chat, split="prompt_only/toxic-chat"
        )
        wandb.log(test_metrics, step=global_step)

        test_metrics_harmbench = self.eval(
            self.test_loader_harmbench, split="with_response/harmbench"
        )
        wandb.log(test_metrics_harmbench, step=global_step)

        test_metrics_openai = self.eval(
            self.test_loader_openai, split="prompt_only/openai-moderation"
        )
        wandb.log(test_metrics_openai, step=global_step)

        test_metrics_wildguard = self.eval(
            self.test_loader_wildguard, split="with_response/wildguard"
        )
        wandb.log(test_metrics_wildguard, step=global_step)


        self.model.save_pretrained(f"save/{self.args.exp_name}")
        self.tokenizer.save_pretrained(f"save/{self.args.exp_name}")

        wandb.finish()

    @torch.no_grad()
    def eval(self, dataloader, split="val"):
        all_preds = []
        all_labels = []
        all_loss = []
        all_scores = []
        self.model.eval()
        for batch in tqdm(
            dataloader, leave=False, dynamic_ncols=True, desc=f"run {split}"
        ):
            chunks = {
                k: torch.chunk(v, self.args.grad_acc_steps, dim=0)
                for k, v in batch.items()
            }
            num_chunks = len(chunks["input_ids"])
            all_labels.append(batch["labels"])
            for i in range(num_chunks):
                mini_batch = {k: v[i].to(self.device) for k, v in chunks.items()}
                try:
                    soft_labels = mini_batch.pop("soft_labels")
                except KeyError:
                    soft_labels = None
                outputs = self.model(**mini_batch)
                loss = outputs.loss
                logits = outputs.logits
                scores = F.softmax(logits, -1)[:, 1]

                all_scores.append(scores.cpu())
                all_loss.append(loss.item())

                preds = torch.argmax(logits, -1).cpu()
                all_preds.append(preds)

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_scores = torch.cat(all_scores, dim=0).numpy()
        avg_loss = np.mean(all_loss)

        precision = precision_score(all_labels, all_preds, zero_division=1.0)
        recall = recall_score(all_labels, all_preds, zero_division=1.0)
        f1 = f1_score(all_labels, all_preds, zero_division=1.0)
        auc = average_precision_score(all_labels, all_scores)

        acc = np.mean(all_preds == all_labels)
        metrics = {
            f"loss/{split}": avg_loss,
            f"f1/{split}": f1,
            f"precision/{split}": precision,
            f"recall/{split}": recall,
            f"auprc/{split}": auc,
            f"acc/{split}": acc,
        }

        return metrics
