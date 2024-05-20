from torch.utils.data import IterableDataset

class StreamedDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, chunk_size=16000):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def __iter__(self):
        return preprocess_function(self.dataset, self.tokenizer, self.chunk_size)

def chunk_text(text, tokenizer, chunk_size=16000):
    tokens = tokenizer(text, return_tensors='pt', max_length=None, truncation=False)['input_ids']
    tokens = tokens.squeeze()
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return chunks


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

            raise ValueError(input_chunks)
            
            input_chunk_tokens = tokenizer(input_chunk, return_tensors='pt', padding="max_length", max_length=chunk_size)
            yield {
                "input_ids": input_chunk_tokens['input_ids'],
                "attention_mask": input_chunk_tokens['attention_mask'],
                "labels": label_tokens.squeeze(),
                "doc_id": idx,
                "chunk_id": chunk_idx
            }
