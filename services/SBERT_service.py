from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.DataProcessing import *

# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def get_embeddings(document):
    embeddings = []
    for sentence in document:
        sentence_vec = get_sentence_embedding(sentence)
        embeddings.append(sentence_vec)
    return embeddings

def get_sentence_embedding(sentence):
    return model.encode(sentence)

def calculate_similarity(text1, text2):
    # Tách và chuẩn hóa câu
    text1_processed = [normalize_sentence(sentence) for sentence in tokenize_sentences(text1)]
    text2_processed = [normalize_sentence(sentence) for sentence in tokenize_sentences(text2)]

    embeddings1 = get_embeddings(text1_processed)
    embeddings2 = get_embeddings(text2_processed)

    # Tính độ tương đồng cosine
    total_similarity = 0

    for embedding1 in embeddings1:
        similarities = [float(cosine_similarity([embedding1], [embedding2])[0][0]) for embedding2 in embeddings2]
        if similarities:
            total_similarity += max(similarities)  # Lấy độ tương đồng cao nhất cho mỗi câu
        else:
            total_similarity += 0
    
    return abs(total_similarity / len(embeddings1)) if embeddings1 else 0