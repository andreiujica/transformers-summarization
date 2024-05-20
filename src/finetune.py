import nltk
nltk.download('punkt')

def segment_sentences(text):
    return nltk.sent_tokenize(text)

def create_chunks(sentences, num_chunks):
    chunk_size = len(sentences) // num_chunks
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(sentences)
        chunk = " ".join(sentences[start_idx:end_idx])
        chunks.append(chunk)
    
    return chunks

def ensure_equal_chunks(document, summary, default_chunks=5):
    document_sentences = segment_sentences(document)
    summary_sentences = segment_sentences(summary)
    num_sentences = len(summary_sentences)
    
    if num_sentences == 1:
        num_sentences = default_chunks

    document_chunks = create_chunks(document_sentences, num_sentences)
    
    return document_chunks, summary_sentences
