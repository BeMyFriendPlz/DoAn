from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.DataProcessing import preprocess_text

word2vec_model = Word2Vec.load(r"C:\Users\B0g3ym4n\Documents\DoAn\model_trainings\word2vec_vi.model")

def get_embeddings(document):
    embeddings = []
    for sentence in document:
        sentence_vec = get_sentence_embedding(sentence)
        embeddings.append(sentence_vec)
    return embeddings

def get_sentence_embedding(sentence):
    words = sentence.split()
    embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0) # trung bình cộng các vector
    else:
        return np.zeros(word2vec_model.vector_size)

def calculate_similarity(text1, text2):
    # Tiền xử lý văn bản
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)

    # Tạo embedding cho từng văn bản
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