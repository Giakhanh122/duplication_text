from HSmodule import *
from sentence_transformers import SentenceTransformer

from ftfy import fix_text
import re

# Chuẩn hóa văn bản
def normalizing(text: str):
        text = fix_text(text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.lower()
        return text

class Shingling:
  def __init__(self, k : int = 5):
    self.k = k
    

  

  def shingle(self, text : str):
    text = normalizing(text)
    shingles = list()
    for i in range(len(text) - self.k + 1):
      shingles.append(text[i:i+self.k])
    return set(shingles)

  def preprocessing(self, texts : list):
        # input list(string) - output list(VectorRecord)
        # k ở đây là n-gram size
        # tạo shingle sets
        shingling = Shingling(self.k)
        shingles_set = [shingling.shingle(t) for t in texts]  #O(n)    n là tổng độ dài toàn bộ văn bản

        # tạo vocab
        vocab = list(set().union(*shingles_set))    #O(n)

        # ánh xạ nhị phân
        vectors = [[1 if x in s else 0 for x in vocab] for s in shingles_set]   #O(m x V)     m là số văn bản, V là số phần tử của vocab
        listVecRecord = [VectorRecord(vec = x, id = id ) for id, x in enumerate(vectors)]       #O(m)

        return listVecRecord
  
  def __call__(self, text : list):
    return self.preprocessing(text)

class TextEmbedder:
  def __init__(self, model_name : str = 'multi-qa-MiniLM-L6-cos-v1'):
    self.model = SentenceTransformer(model_name)
    self.lenOfVector = self.model.get_sentence_embedding_dimension()

  def preprocessing(self, texts : list):
        # input list(string) - output list(VectorRecord)
        # chuẩn hóa văn bản
        texts = [normalizing(t) for t in texts]

        embeddings = self.model.encode(texts)   #O(n)    n là tổng độ dài toàn bộ văn bản
        listVecRecord = [VectorRecord(vec = emb, id = id ) for id, emb in enumerate(embeddings)]   #O(m)

        return listVecRecord
  
  def __call__(self, text : list):
    return self.preprocessing(text)
