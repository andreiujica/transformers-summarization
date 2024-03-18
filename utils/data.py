from datasets import load_dataset

def get_dataset_sample(category="g", num_samples=10):
    """
    This function loads a small set of samples from the BIGPATENT dataset for demonstration and initial setup.
    """
    dataset = load_dataset("big_patent", category, split='test', trust_remote_code=True)
    samples = dataset.select(range(num_samples))
    return [{"text": sample["description"], "summary": sample["abstract"]} for sample in samples]