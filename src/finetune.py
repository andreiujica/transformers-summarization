from torch.utils.data import IterableDataset
from transformers import DataCollatorForSeq2Seq

class StreamedDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, chunk_size=16000):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def __iter__(self):
        return preprocess_function(self.dataset, self.tokenizer, self.chunk_size)

def chunk_text(text, tokenizer, chunk_size=16000):
    tokens = tokenizer(text, return_tensors='pt', max_length=None, truncation=False)['input_ids']
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return chunks


class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # Extract doc_ids and chunk_ids
        doc_ids = [feature.pop('doc_id') for feature in features]
        chunk_ids = [feature.pop('chunk_id') for feature in features]
        
        # Use the parent class to collate the remaining fields
        batch = super().__call__(features)
        
        # Add doc_ids and chunk_ids back to the batch
        batch['doc_ids'] = doc_ids
        batch['chunk_ids'] = chunk_ids
        
        return batch


def preprocess_function(examples, tokenizer, chunk_size=16000):
    inputs = examples['description']
    abstracts = examples['abstract']

    if not inputs or not abstracts:
        raise ValueError("Inputs or abstracts are empty")

    for idx, (input_text, abstract) in enumerate(zip(inputs, abstracts)):

        input_chunks = chunk_text(input_text, tokenizer, chunk_size)
        label_tokens = tokenizer(abstract, return_tensors='pt', max_length=2048, truncation=True, padding="max_length")['input_ids']

        if not input_chunks:
            raise ValueError(f"No chunks generated for document {idx}") 
        
        for chunk_idx, input_chunk in enumerate(input_chunks):

            input_chunk_tokens = tokenizer.pad(
                {'input_ids': input_chunk}, 
                padding="max_length", 
                max_length=chunk_size, 
                return_tensors='pt'
            )
            yield {
                "input_ids": input_chunk_tokens['input_ids'].squeeze(0),
                "attention_mask": input_chunk_tokens['attention_mask'].squeeze(0),
                "labels": label_tokens.squeeze(0),
                "doc_id": idx,
                "chunk_id": chunk_idx
            }
