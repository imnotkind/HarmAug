import gzip
import heapq
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List

import editdistance
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from vllm import LLM, SamplingParams


class RobertaClassifier(object):
    def __init__(self, device) -> None:
        self.device = device
        self.toxicity_fn = pipeline("text-classification",
                                    model="facebook/roberta-hate-speech-dynabench-r4-target",
                                    device=device,
                                    )

    def compute(self, predictions):
        def _get_score(x):
            if x["label"] == "hate":
                return x["score"]
            else:
                return 1 - x["score"]  # 1 - nothate
        outputs = self.toxicity_fn(predictions, truncation=True)
        toxicity_scores = list(map(lambda x: _get_score(x), outputs))
        return toxicity_scores



class VLLMGuard(object):
    def __init__(self, device, version=3, pbar=False) -> None:
        if version == 1:
            model_id = "meta-llama/LlamaGuard-7b"
            max_model_len = 4096
        elif version ==2:
            print("version-2")
            model_id = "meta-llama/Meta-Llama-Guard-2-8B"
            max_model_len = 8192
        else:
            print("version-3")
            model_id = "meta-llama/Llama-Guard-3-8B"
            max_model_len = 8192
        
        self.version = version
        dtype = "bfloat16"
        self.pbar = pbar

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(model_id, dtype=dtype,
                       max_model_len=max_model_len,
                       device=device
                       )
            

        self.sampling_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=2)

        unsafe_token = self.tokenizer.tokenize("unsafe")
        safe_token = self.tokenizer.tokenize("safe")
        
        self.unsafe_id = self.tokenizer.convert_tokens_to_ids(unsafe_token)[0]
        self.safe_id = self.tokenizer.convert_tokens_to_ids(safe_token)[0]
        
    def make_chat(self, prompts, responses=None):
        if responses is not None:
            assert len(prompts) == len(responses)
            chats = []
            for x,y in zip(prompts, responses):
                if y is not None:
                    chat = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": x},
                        {"role": "assistant", "content": y},
                    ],
                    tokenize=False,
                )
                else:
                    chat = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": x},
                    ],
                    tokenize=False,
                )
                chats.append(chat)
            assert len(chats) == len(prompts) & len(chats) == len(responses)
        else:
            chats = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": x}], tokenize=False
                )
                for x in prompts
            ]
        if self.version == 3:
            chats = [chat +"\n\n" for chat in chats]
        return chats
        

    def compute(self, prompts, responses=None):
        system_prompts = self.make_chat(prompts, responses)
        sampling_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=2)
        
        llm_outputs = self.llm.generate(
            system_prompts, sampling_params, use_tqdm=self.pbar)
        log_probs = []

        for output in llm_outputs:
            log_prob = output.outputs[0].logprobs[0][self.unsafe_id].logprob
            log_probs.append(log_prob)
        log_probs = torch.tensor(log_probs)
        
        return log_probs

    def get_soft_label(self, prompts, responses=None, temperature=1.0):
        system_prompts = self.make_chat(prompts, responses)
        sampling_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=2)
        sampling_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=2)
        
        llm_outputs = self.llm.generate(
            system_prompts, sampling_params, use_tqdm=self.pbar)
        
        logits = []

        for output in llm_outputs:
            safe_log_prob = output.outputs[0].logprobs[0][self.safe_id].logprob
            unsafe_log_prob = output.outputs[0].logprobs[0][self.unsafe_id].logprob
            log_probs = [safe_log_prob, unsafe_log_prob]
            logits.append(log_probs)
        logits = torch.tensor(logits, dtype=torch.float)
        soft_labels = F.softmax(logits / temperature, -1)
        unsafe_labels = soft_labels[:, 1]
        unsafe_log_probs = torch.log(unsafe_labels)
        
        return unsafe_log_probs

    def generate_label(self, prompts, responses=None):
        chats = self.make_chat(prompts, responses)
        sampling_params = SamplingParams(temperature=0, max_tokens=20, min_tokens=1)
        llm_outputs = self.llm.generate(chats, sampling_params, use_tqdm=self.pbar)
        labels = []

        for output in llm_outputs:
            label = output.outputs[0].text
            labels.append(label)
        
        return labels
    

class LlamaToxicClassifier(object):
    def __init__(self, device, version=3) -> None:
        if version == 1:
            model_id = "meta-llama/LlamaGuard-7b"
        elif version == 2:
            model_id = "meta-llama/Meta-Llama-Guard-2-8B"
        elif version == 3:
            model_id = "meta-llama/Llama-Guard-3-8B"
        elif version == "1b":
            model_id = "meta-llama/Llama-Guard-3-1B"
        
        dtype = "bfloat16"

        self.version = version
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.device,
            # attn_implementation="flash_attention_2",
        )
        self.model.eval()
        unsafe_token = self.tokenizer.tokenize("unsafe")
        self.unsafe_id = self.tokenizer.convert_tokens_to_ids(unsafe_token)[0]

        if version == 3:
            assert len(self.tokenizer.encode("\n\n")) == 1
            self.newline_id = self.tokenizer.encode("\n\n")[0]
        elif version == "1b":
            self.newline_id = 271
    
    def make_chat_v2(self, prompts, responses=None):
        if responses is not None:
            assert len(prompts) == len(responses)
            chats = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": [{"type":"text", "text":x}]},
                        
                        {"role": "assistant", "content": [{"type":"text", "text":y}]},
                    ],
                    tokenize=False,
                )
                for x, y in zip(prompts, responses)
            ]
        else:
            chats = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": [{"type":"text", "text":x}]}], tokenize=False
                )
                for x in prompts
            ]

        return chats
    
    def make_chat(self, prompts, responses=None):
        if responses is not None:
            assert len(prompts) == len(responses)
            chats = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": x},
                        {"role": "assistant", "content": y},
                    ],
                    tokenize=False,
                )
                for x, y in zip(prompts, responses)
            ]
        else:
            chats = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": x}], tokenize=False
                )
                for x in prompts
            ]

        return chats
    @torch.no_grad()
    def compute(self, prompts, responses=None):
        if self.version == "1b":
            chats = self.make_chat_v2(prompts, responses)
        else:
            chats = self.make_chat(prompts, responses)
        inputs = self.tokenizer(chats, padding=True, return_tensors="pt")

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Llama-guard-3 predict safe/unsafe after the token "\n\n"
        if self.version == 3 or self.version == "1b":
            bs = input_ids.size(0)
            _new_id = (
                torch.ones((bs, 1), dtype=torch.long, device=self.device)
                * self.newline_id
            )
            _new_mask = torch.ones_like(_new_id)
            input_ids = torch.cat([input_ids, _new_id], dim=1)
            attention_mask = torch.cat([attention_mask, _new_mask], dim=1)

        self.model.eval()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        log_probs = F.log_softmax(logits, -1)
        log_rewards = log_probs[:, self.unsafe_id]

        return log_rewards

    def generate_label(self, prompts, responses=None):
        chats = self.make_chat(prompts, responses)
        inputs = self.tokenizer(chats, padding=True, return_tensors="pt")

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Llama-guard-3 predict safe/unsafe after the token "\n\n"
        if self.version == 3:
            bs = input_ids.size(0)
            _new_id = (
                torch.ones((bs, 1), dtype=torch.long, device=self.device)
                * self.newline_id
            )
            _new_mask = torch.ones_like(_new_id)
            input_ids = torch.cat([input_ids, _new_id], dim=1)
            attention_mask = torch.cat([attention_mask, _new_mask], dim=1)

        prompt_len = inputs["input_ids"].size(1)
        outputs = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=20)

        labels = self.tokenizer.batch_decode(
            outputs[:, prompt_len:], 
            skip_special_tokens=True)        
        
        return labels
        

class StudentModel(object):
    def __init__(self, ckpt, device) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
    
    def predict(self, prompts):
        inputs = self.tokenizer(
            prompts, padding=True, return_tensors="pt", truncation=True)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))
        logits = outputs.logits
        preds = torch.argmax(logits, 1)

        return preds


def batch_cosine_similarity_kernel(embeddings, batch_size=16):
    num_samples = embeddings.size(0)
    avg_sim = 0.0

    for i in tqdm(range(0, num_samples, batch_size)):
        batch_end = min(i + batch_size, num_samples)
        batch = embeddings[i:batch_end, :]
        with torch.no_grad():
            cos_sim_batch = F.linear(F.normalize(batch), F.normalize(embeddings))
        avg_sim += cos_sim_batch.sum().item()

    # Adjust for duplicate pairs and remove diagonal components
    diag = 0.0
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch = embeddings[i:batch_end, :]
        diag += F.cosine_similarity(batch, batch, dim=-1).sum().item()
    avg_sim -= diag

    # Compute average similarity
    avg_sim /= num_samples * (num_samples - 1)

    return avg_sim


def seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names(model) -> List[str]:
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}


class InfIterator(object):
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)

    def __len__(self):
        return len(self.iterator)


def lora_to_base(model):
    try:
        model.base_model.disable_adapter_layers()
    except:
        print("No adapter layers to disable")
    model.eval()


def base_to_lora(model):
    try:
        model.base_model.enable_adapter_layers()
    except:
        print("No adapter layers to enable")
    model.train()


@dataclass(order=True)
class TrajectoryWithReward:
    prompt_ids: torch.LongTensor = field(compare=False)
    c_log_reward: float = field(compare=False)
    lm_log_reward: float = field(compare=False)
    log_reward: float = field(compare=True)  # sorting based on this
    prompt: str = field(compare=False)
    emb: torch.tensor = field(compare=False)
    ref_reward: float = field(compare=False, init=False)

    def __post_init__(self):
        self.ref_reward = self.log_reward


@dataclass(order=True)
class TrajectoryWithCReward:
    prompt_ids: torch.LongTensor = field(compare=False)
    c_log_reward: float = field(compare=True)  # sorting based on this
    lm_log_reward: float = field(compare=False)
    log_reward: float = field(compare=False)
    prompt: str = field(compare=False)
    emb: torch.tensor = field(compare=False)
    ref_reward: float = field(compare=False, init=False)

    def __post_init__(self):
        self.ref_reward = self.c_log_reward


class ReplayBuffer(object):
    def __init__(
        self,
        eos_token_id,
        max_size=1000,
        sim_tolerance=0.25,
        prioritization="c_reward",
        compare="reward",
    ):
        self.eos_token_id = eos_token_id
        self.max_size = max_size
        self.sim_tolerance = sim_tolerance
        self.buffer = []
        self.prompt_pool = set()
        self.prioritization = prioritization
        self.compare = compare

        if compare == "c_reward":
            print("comparison with c_reward")
            self.Trajectory = TrajectoryWithCReward
        else:
            print("comparison with total reward")
            self.Trajectory = TrajectoryWithReward

    def size(self):
        return len(self.buffer)

    def add(self, item):
        # check whether the item has been already added before.
        if item.prompt in self.prompt_pool:
            return
        tokens = [x for x in item.prompt_ids.tolist() if x != self.eos_token_id]
        # find examples that are similar to the item and replace it with new one if new one has higher reward
        for buffer_item in self.buffer:
            existing_tokens = [
                x for x in buffer_item.prompt_ids.tolist() if x != self.eos_token_id
            ]
            if (
                editdistance.eval(tokens, existing_tokens)
                < (len(tokens) + len(existing_tokens)) * self.sim_tolerance
            ):
                if buffer_item.ref_reward >= item.ref_reward:
                    return
                else:
                    # remove the old item
                    self.prompt_pool.discard(buffer_item.prompt)
                    self.buffer.remove(buffer_item)
                    heapq.heapify(self.buffer)

                    # add new item
                    self.prompt_pool.add(item.prompt)
                    heapq.heappush(self.buffer, item)

                    if len(self.buffer) != len(self.prompt_pool):
                        self.prompt_pool = set([x.prompt for x in self.buffer])
                    return

        self.prompt_pool.add(item.prompt)

        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, item)
        else:
            popped = heapq.heappushpop(self.buffer, item)
            try:
                self.prompt_pool.remove(popped.prompt)
            except KeyError:
                self.prompt_pool = set([x.propt for x in self.buffer])

    def add_batch(
        self, prompt_ids, prompts, embs, c_log_rewards, lm_log_rewards, log_rewards
    ):
        # move tensors to cpu
        prompt_ids = prompt_ids.cpu()
        embs = embs.cpu()

        pad_mask = (prompt_ids == self.eos_token_id).cumsum(1) > 1
        attention_mask = (~pad_mask).long()
        lengths = torch.sum(attention_mask, 1)

        for i in range(log_rewards.size(0)):
            c_log_reward = c_log_rewards[i].item()
            lm_log_reward = lm_log_rewards[i].item()
            log_reward = log_rewards[i].item()
            length = lengths[i].item()
            emb = embs[i]
            # add new item
            item = self.Trajectory(
                prompt_ids[i, :length],
                c_log_reward,
                lm_log_reward,
                log_reward,
                prompts[i],
                emb,
            )

            self.add(item)

    def sample(self, num_samples):
        if self.prioritization == "reward":
            priorities = [item.log_reward for item in self.buffer]
            priorities = np.array(priorities)
            priorities = priorities - np.max(priorities)
            priorities = np.exp(priorities)
            prob = priorities / np.sum(priorities)

        elif self.prioritization == "c_reward":
            priorities = [item.c_log_reward for item in self.buffer]
            priorities = np.array(priorities)
            priorities = priorities - np.max(priorities)
            priorities = np.exp(priorities)
            prob = priorities / np.sum(priorities)

        elif self.prioritization == "uniform":
            prob = np.ones(len(self.buffer)) / len(self.buffer)

        idx = np.random.choice(len(self.buffer), num_samples, p=prob, replace=False)

        # right-side padding
        prompts = [self.buffer[i].prompt for i in idx]
        prompt_ids = [self.buffer[i].prompt_ids for i in idx]
        attention_mask = [torch.ones_like(x) for x in prompt_ids]

        prompt_ids = pad_sequence(
            prompt_ids, batch_first=True, padding_value=self.eos_token_id
        )
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        batch = {
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
            "prompts": prompts,
        }

        c_log_rewards = torch.tensor([self.buffer[i].c_log_reward for i in idx])
        lm_log_rewards = torch.tensor([self.buffer[i].lm_log_reward for i in idx])
        log_rewards = torch.tensor([self.buffer[i].log_reward for i in idx])

        reward_batch = {
            "c_log_reward": c_log_rewards,
            "lm_log_reward": lm_log_rewards,
            "log_reward": log_rewards,
        }

        return batch, reward_batch

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with gzip.open(path, "rb") as f:
            self.buffer = pickle.load(f)
        heapq.heapify(self.buffer)


class CosineRelayBuffer(ReplayBuffer):
    def __init__(
        self,
        eos_token_id,
        max_size=1000,
        sim_tolerance=0.4,
        prioritization="c_reward",
        compare="reward",
    ):
        super().__init__(eos_token_id, max_size, sim_tolerance, prioritization, compare)

    def add(self, item):
        # check whether the item has been already added before.
        if item.prompt in self.prompt_pool:
            return

        if len(self.buffer) > 0:
            buffer_embs = torch.stack(
                [item.emb for item in self.buffer], dim=0
            )  # [b,d]
            # find examples that are similar to the item and replace it with new one if new one has higher reward
            query = item.emb.unsqueeze(0)  # [1,d]
            cos_sims = F.cosine_similarity(query, buffer_embs, dim=1)
            max_id = torch.argmax(cos_sims, dim=0)
            max_sim = cos_sims[max_id].item()

            if max_sim > self.sim_tolerance:
                buffer_item = self.buffer[max_id]
                if buffer_item.ref_reward >= item.ref_reward:
                    return
                else:
                    self.prompt_pool.discard(buffer_item.prompt)
                    self.buffer.remove(buffer_item)
                    heapq.heapify(self.buffer)

                    # add new item
                    self.prompt_pool.add(item.prompt)
                    heapq.heappush(self.buffer, item)

                    if len(self.buffer) != len(self.prompt_pool):
                        self.prompt_pool = set([x.prompt for x in self.buffer])
                    return

        self.prompt_pool.add(item.prompt)

        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, item)
        else:
            popped = heapq.heappushpop(self.buffer, item)
            try:
                self.prompt_pool.remove(popped.prompt)
            except KeyError:
                self.prompt_pool = set([x.prompt for x in self.buffer])
