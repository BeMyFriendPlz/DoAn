from datasets import load_dataset

# Tải dataset
dataset = load_dataset("thanhdath/vietnamese-sentences", split="train")

# Tổng số câu cần lấy
total_samples = 1000000  # 1.000.000 câu

# Lấy ngẫu nhiên 1.000.000 câu
train_data = dataset.shuffle(seed=42).select(range(total_samples))

# Lưu tập train vào file
with open(r'C:\Users\B0g3ym4n\Documents\DoAn\data\train_data.txt', 'w', encoding='utf-8') as train_file:
    for sentence in train_data['text']:
        train_file.write(sentence + '\n')

print("Đã lưu tập train (1.000.000 câu) vào 'train_data.txt'.")

