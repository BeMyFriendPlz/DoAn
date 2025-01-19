import os, concurrent.futures
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from pathlib import Path
from utils.DataProcessing import process_file

# Sử dụng os.path.join để tạo đường dẫn tương đối
base_directory = os.path.dirname(__file__)
data_directory = os.path.join(base_directory, "data")
data_output_path = os.path.join(base_directory, "dataset.txt")
model_directory = os.path.join(base_directory, "model_trainings")
word2vec_model_path = os.path.join(model_directory, "word2vec_vi.model")
doc2vec_model_path = os.path.join(model_directory, "doc2vec_vi.model")

def generate_dataset():
    if not os.path.exists(data_output_path):
        folder = Path(data_directory)

        file_paths = list(folder.rglob("*.txt"))  # Lấy danh sách file
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_file, file_paths)

        print(f"Dữ liệu đã xử lý thành công và lưu tại '{data_output_path}'")
    else:
        print("Dữ liệu đã xử lý đã tồn tại, không cần xử lý lại.")

# Tạo và xử lý dữ liệu dataset
generate_dataset()

# Hàm đọc dữ liệu văn bản đã xử lý
def read_dataset_for_Word2Vec():
    with open(data_output_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.rstrip().split()

def read_dataset_for_Doc2Vec():
    with open(data_output_path, "r", encoding="utf-8") as file:
       for i, line in enumerate(file):
            sentence = line.rstrip().split()
            yield TaggedDocument(words=sentence, tags=[f'SEN_{i}'])

# Chuẩn bị dữ liệu cho Word2Vec và Doc2Vec
word2vec_sentences = list(read_dataset_for_Word2Vec())
doc2vec_documents = list(read_dataset_for_Doc2Vec())
print("Chuẩn bị dữ liệu thành công!")

# Huấn luyện và lưu mô hình Word2Vec nếu chưa tồn tại
if not os.path.exists(word2vec_model_path):
    print("Đang huấn luyện mô hình Word2Vec...")
    word2vec_model = Word2Vec(sentences=word2vec_sentences, vector_size=300, window=10, min_count=5, workers=8, sg=0, epochs=20)
    word2vec_model.save(word2vec_model_path)
    print(f"Đã lưu mô hình Word2Vec tại '{word2vec_model_path}'")
else:
    print("Mô hình Word2Vec đã tồn tại, không cần huấn luyện lại.")

# Huấn luyện và lưu mô hình Doc2Vec nếu chưa tồn tại
if not os.path.exists(doc2vec_model_path):
    print("Đang huấn luyện mô hình Doc2Vec...")
    doc2vec_model = Doc2Vec(documents=doc2vec_documents, vector_size=300, window=10, min_count=5, workers=8, dm=1, epochs=20)
    doc2vec_model.save(doc2vec_model_path)
    print(f"Đã lưu mô hình Doc2Vec tại '{doc2vec_model_path}'")
else:
    print("Mô hình Doc2Vec đã tồn tại, không cần huấn luyện lại.")