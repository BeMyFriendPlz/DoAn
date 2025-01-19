from transformers import AutoModel, AutoTokenizer
import torch
from utils.DataProcessing import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity

# Sử dụng phiên bản PhoBERT: "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModel.from_pretrained("vinai/phobert-base-v2")

def get_embeddings(document):
    embeddings = []
    for sentence in document:
        sentence_vec = get_sentence_embedding(sentence)
        embeddings.append(sentence_vec)
    return embeddings

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    
    # Mean pooling
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).squeeze().numpy()
    return embedding

# Hàm tính toán độ tương đồng giữa hai văn bản
def calculate_similarity(text1, text2):
    # Tiền xử lý văn bản
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)

    # Tạo embedding cho mỗi văn bản
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