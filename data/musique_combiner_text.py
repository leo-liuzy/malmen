from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig

import torch

from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

from data.base import BaseDataset
from transformers import AutoTokenizer
from knowledge_propagation.utils import io
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math

class MUSIQUE_COMBINER_TEXTDataset(BaseDataset):
    def __init__(
        self,
        config: DictConfig,
        path: str,
        tok: AutoTokenizer,
        device: Union[int, str, torch.device]
    ):

        self.config = config
        self.data = io.load_jsonlines(path)

        from data.nq import NQDataset
        self.nq = NQDataset(self.config.nq_path + ("/train.json" if "train" in path else "/validation.json"), tok, config)
        
        self.tok = tok
        self.device = device
        self.entry2microBS = None
        self.show_first_example = False
    
    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        assert len(row["texts"]) == 2
        texts = row["texts"]
        unrel_prompt = self.nq[idx][0] + "?"
        unrel_answer = self.nq[idx][1]
        
        tuples = []
        for q in row["multi_hop_efficacy"]:
            tuples.append(self.tok_tuples(q["question"], q["answer"]))
        
        if self.config.outer_loop_include_atomq:
            for q in row["single_hop_efficacy"]:
                tuples.append(self.tok_tuples(q["question"], q["answer"]))
        
        np.random.shuffle(tuples)
        np.random.shuffle(texts)
        ret = {
            # inner loop
            "edit_tuples": self.tok_texts(texts),
            # outer loop
            "equiv_tuples": self.pad_tok_tuples(tuples),
            # locality
            "unrel_tuples": self.tok_tuples(unrel_prompt, unrel_answer)
        }
        if self.entry2microBS is None:
            self.entry2microBS = {
                k: v["input_ids"].shape[0]
                for k, v in ret.items()
            }
        if not self.show_first_example:
            for k, v in ret.items():
                print(f"{k}: ")
                print(f"input_ids: " + "\n@@\n".join(self.tok.batch_decode(v["input_ids"])))
            self.show_first_example = True
        
        return ret
        
    def tok_tuples(
        self,
        prompt: str,
        answer: str
    ) -> Dict[str, torch.LongTensor]:

        assert isinstance(self.tok, GPT2TokenizerFast) or "llama-3.2" in self.tok.name_or_path.lower()
        
        answer = " " + answer
            
        tok_prompt = self.tok(
            prompt,
            return_tensors = "pt",
        )
        
        tok_answer = {
            k: torch.concat(
                [
                    v,
                    torch.full(
                        (v.shape[0], 1), # shape of the constant tensor
                        (
                            1 
                            if k == "attention_mask" else
                            self.tok.eos_token_id # this is to teach the model to end after outputing the answer.
                        )
                    )
                ], dim=-1
            )
            for k, v in self.tok(
                answer,
                return_tensors = "pt",
                add_special_tokens = False
            ).items()
        }
        

        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)
        
        return tok_tuples
    
    def tok_texts(
        self,
        texts: List[str],
    ) -> Dict[str, torch.LongTensor]:
        
        ret = {
            f"{k}": 
                torch.concat(
                    [
                        v, 
                        torch.full(
                            (v.shape[0], 1), # shape of the constant tensor
                            (
                                1 
                                if k == "attention_mask" else
                                self.tok.eos_token_id # this is to teach the model to end after outputing the answer.
                            )
                        )
                    ], dim=-1)
            for k, v in self.tok(
                texts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=True, # make the SFT label free of BOS
            ).items()
        }
        # ret["input_ids"][ret["input_ids"] == self.tok.pad_token_id] = -100
        input_ids = ret["input_ids"].clone()[:, :-1]
        labels = ret["input_ids"].clone()[:, 1:]
        labels[labels == self.tok.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": ret["attention_mask"][:, :-1],
            "labels": labels
        }
    
    def pad_tok_tuples(
        self,
        tok_tuples: List[Dict[str, torch.LongTensor]]
    ) -> Dict[str, torch.LongTensor]:
        
        return {
            k: pad_sequence(
                [e for t in tok_tuples for e in t[k]],
                batch_first = True,
                padding_value = -100 if k == "labels" else self.tok.pad_token_id,
                padding_side="left"
            ).to(self.device)
            for k in tok_tuples[0].keys()
        }
    
    def collate_fn(
        self,
        tuples: Tuple[Dict[str, Dict[str, torch.LongTensor]]]
    ) -> Dict[str, List[Dict[str, torch.LongTensor]]]:
        
        tuples: Dict[str, List[Dict[str, torch.LongTensor]]] = {
            k: sorted(
                [t[k] for t in tuples],
                key = lambda x: x["attention_mask"].sum().item(),
                reverse = True
            )
            for k in tuples[0].keys()
        }
        assert self.entry2microBS is not None
        return {
            k: [
                self.pad_tok_tuples(
                    v[n_batch * self.config.batch_size
                      :(n_batch + 1) * self.config.batch_size]
                )
                for n_batch in range(math.ceil(self.config.n_edits / self.config.batch_size))
            ]
            for k, v in tuples.items()
        }