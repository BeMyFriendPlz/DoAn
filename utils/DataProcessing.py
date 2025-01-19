import py_vncorenlp, re, nltk, unicodedata, underthesea
from tqdm import tqdm

rdrsegmenter = None

def get_rdrsegmenter():
    global rdrsegmenter
    if rdrsegmenter is None:
        rdrsegmenter = py_vncorenlp.VnCoreNLP(
            save_dir=r'C:\Users\B0g3ym4n\Documents\DoAn\VnCoreNLP',
            annotators=["wseg"]
        )
    return rdrsegmenter

# Hàm để đọc file stopwords
def stopwords():
    with open(r"C:\Users\B0g3ym4n\Documents\DoAn\vietnamese-stopwords.txt", encoding='utf-8') as f:
        return set([line.strip() for line in f])

# Hàm để tách câu từ nội dung
def tokenize_sentences(content):
    sentences = nltk.sent_tokenize(content)
    return sentences

# Hàm để chuẩn hóa unicode thành unicode normative form (NFC)
def standardize_unicode(sentence):
    sentence = unicodedata.normalize('NFC', sentence)
    return sentence

# Hàm để tokenizer từ sử dụng vncorenlp
def tokenize_text(text):
    tokens = get_rdrsegmenter().word_segment(text)
    return tokens

# Hàm để loại bỏ các ký tự đặc biệt và dấu câu trong câu
def normalize_sentence(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence).strip().lower()    # Loại bỏ các ký tự đặc biệt và dấu câu
    sentence = re.sub(r'\s+', ' ', sentence)    # Loại bỏ khoảng trắng thừa
    sentence = standardize_unicode(sentence)
    sentence = underthesea.text_normalize(sentence)
    return sentence

# Hàm để chuẩn hóa từ và loại bỏ ký tự đặc biệt cũng như dấu cách thừa
def normalize_word(word):
    normalized_word = re.sub(r'[^\w\s]', '', word.strip()) # Loại bỏ ký tự đặc biệt
    normalized_word = normalized_word.lower() # Chuyển từ thành viết thường
    return normalized_word

# Hàm để loại bỏ các từ là số và các link
def is_valid_word(word):
    return (not word.isdigit()
            and not re.search(r'http', word)
            and not re.search(r'www', word)
            and any(c.islower() for c in word)
            and not any(c.isdigit() for c in word)
            and word not in stopwords()
            )

def write_to_file(sentences):
    with open(r"C:\Users\B0g3ym4n\Documents\DoAn\dataset.txt", 'a', encoding='utf-8') as out_file:
        for sentence in sentences:
            out_file.write(sentence + '\n')

def preprocess_text(text):

    preprocessText = []

    # Tách câu từ file
    sentences = tokenize_sentences(text)
    for sentence in sentences:
        # Chuẩn hóa câu
        sentence = normalize_sentence(sentence)
        # Tokenizer các từ trong câu sử dụng vncorenlp
        tokens = tokenize_text(sentence)
        if not tokens:  # Kiểm tra nếu tokens là danh sách rỗng
            continue
        valid_tokens = []
        for word in (tokens[0]).split(' '):
            # Loại bỏ các từ không hợp lệ và chuẩn hóa từng từ
            if is_valid_word(word):
                valid_tokens.append(normalize_word(word))
        # Thêm các từ hợp lệ vào dataset
        preprocessText.append(" ".join(valid_tokens))

    return preprocessText

def open_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def process_file(file_path, output_path = r"C:\Users\B0g3ym4n\Documents\DoAn\dataset.txt"):
    with open(file_path, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)
    with open(file_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Processing", unit="line"):
            processed_lines = preprocess_text(line)  # Xử lý từng dòng
            for processed_line in processed_lines:
                outfile.write(processed_line + "\n")

    print(f"Processed and saved to: {output_path}")

    # text = open_file(file_path)  # Đọc nội dung file
    # processed_text = preprocess_text(text)  # Xử lý nội dung file
    # write_to_file(processed_text)  # Ghi nội dung đã xử lý
    # print(f"Processed: {file_path}")
