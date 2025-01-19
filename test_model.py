from utils.DataProcessing import *
import services.Word2Vec_service as w2v
import services.Doc2Vec_service as d2v
import services.PhoBERT_service as phoBERT
import services.SBERT_service as sBERT

text_1 = "Là học sinh PTIT."
text_2 = "Là sinh viên PTIT."

print(f"Word2Vec - Kết quả độ tương đồng là: {w2v.calculate_similarity(text_1, text_2)}")
print(f"Doc2Vec - Kết quả độ tương đồng là: {d2v.calculate_similarity(text_1, text_2)}")
print(f"PhoBERT - Kết quả độ tương đồng là: {phoBERT.calculate_similarity(text_1, text_2)}")
print(f"SBERT - Kết quả độ tương đồng là: {sBERT.calculate_similarity(text_1, text_2)}")