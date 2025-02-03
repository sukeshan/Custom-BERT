import random

import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class DataLoader:
    
    """DataLoader class for handling batch processing."""

    def __init__(self, batch_size: int = 3, data: list[str] = None, labels: list[int] = None):
       
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        self.batch_size = batch_size
        self.data = data or []
        self.labels = labels or []
        self.encoding = self.tokenizer(self.data)

    def process_data(self):

        """Preprocess data for efficient batching."""

        # Sort by input_ids length to minimize padding
        self.sorted_data = sorted(list(zip(self.encoding['input_ids'], self.encoding['attention_mask'], self.labels)),
                                  key=lambda x: len(x[0])
                                )
        
    def __len__(self):
        return len(self.sorted_data)
    
    def __iter__(self):

        """Iterate over the DataLoader object."""
        
        while len(self.sorted_data) > 0:

            current_batch_size = min(self.batch_size, len(self.sorted_data))
            randn = random.randint(0, max(0, len(self.sorted_data) - current_batch_size))
            
            batch_data = self.sorted_data[randn : current_batch_size + randn]
            self.batch_ids, self.batch_attention_mask, self.batch_labels = zip(*batch_data)

            padded_inputs = pad_sequence(
                    [torch.tensor(ids, dtype=torch.long) for ids in self.batch_ids],
                    batch_first=True,
                    padding_value=0
                )
            padded_masks = pad_sequence(
                    [torch.tensor(mask, dtype=torch.long) for mask in self.batch_attention_mask],
                    batch_first=True,
                    padding_value=0
                )
                    
            del self.sorted_data[randn : current_batch_size + randn ]

            yield {"input_ids" : padded_inputs, 
                "attention_mask" : padded_masks,
                "labels" : torch.tensor(self.batch_labels)
                }
