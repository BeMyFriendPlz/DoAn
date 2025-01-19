import matplotlib.pyplot as plt

word_count = {}  # Khởi tạo dictionary rỗng

# Đọc file txt
with open(r'C:\Users\B0g3ym4n\Documents\DoAn\data\train_data.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    lines = text.split('\n')
    for line in lines:
        if line.strip():  # Đảm bảo dòng không rỗng hoặc chỉ chứa khoảng trắng
            words = line.split()
            word_length = len(words)
            if word_length not in word_count:
                word_count[word_length] = 1  # Nếu chưa có key, khởi tạo giá trị là 1
            else:
                word_count[word_length] += 1  # Nếu đã có key, tăng giá trị lên 1

# Tách keys và values thành danh sách
keys = list(word_count.keys())
values = list(word_count.values())

# Vẽ biểu đồ
plt.bar(keys, values, color='skyblue')

# Thêm nhãn và tiêu đề
plt.xlabel('Số lượng từ trong câu', fontsize=14)
plt.ylabel('Số lượng câu', fontsize=14)
plt.title('Phân phối số lượng từ trong câu', fontsize=16)

# Hiển thị biểu đồ
plt.show()