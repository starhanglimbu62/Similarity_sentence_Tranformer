import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ['This is an example sentence', 'Each sentence is converted']

embeddings = model.encode(sentences)
print("Sentence embeddings:")
print(embeddings.shape)

cos = util.cos_sim(embeddings, embeddings)

print("The similarity between the two sentences is:", cos[0][1])



