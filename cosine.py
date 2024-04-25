# RAG Model
import pickle
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
load1 = PyPDFLoader("C:/Users/ASUS/OneDrive/Desktop/dataset/interview.pdf")
document = load1.load()
text_splitter = CharacterTextSplitter(chunk_size=500)
split_docy = text_splitter.split_documents(document)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="your_api_key")
vectordata = FAISS.from_documents(split_docy, embeddings)
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
retriever = vectordata.as_retriever(search_type="similarity")
rqa = RetrievalQA.from_chain_type(llm=OpenAI( openai_api_key="your_api_key"),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=False)
import numpy as np
bot_answer = "test your chatbot and give the answer provided by your bot."
manual_answer = "Ask the same question to your friends and give that answer."



def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    if magnitude1 == 0 or magnitude2 == 0:
    # Avoid division by zero
        return 0
  
    return dot_product / (magnitude1 * magnitude2)

# Preprocess text (replace with your preferred method)
def preprocess_text(text):
     # Convert to lowercase, remove punctuation, tokenize
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    words = text.split()
    return words
 

def calculate_idf(word_counts, all_words, total_documents):
    document_frequency = sum(1 for count in word_counts.values() if count > 0)
    if document_frequency == 0:
        return 0

  # Count the number of documents containing the word
 
  # Avoid division by zero (replace with smoothing techniques if needed)
  
  
    return np.log(total_documents / document_frequency)

from collections import Counter
def tfidf_vectorize(text, all_words, word_counts_all_documents, total_documents):
    word_counts = Counter(preprocess_text(text))
    tf_vector = np.zeros(len(all_words))
    for i, word in enumerate(all_words):
        tf_vector[i] = word_counts.get(word, 0) / len(text)  # Term Frequency
  
    idf_vector = np.ones(len(all_words))
    for i, word in enumerate(all_words):
        idf_vector[i] = calculate_idf(word_counts_all_documents, all_words, total_documents)
  
    return tf_vector * idf_vector  # TF-IDF vector
documents = [bot_answer, manual_answer]  # Example document collection
total_documents = len(documents)
word_counts_all_documents = Counter()
for doc in documents:
    word_counts_all_documents.update(preprocess_text(doc))
all_words = set(word_counts_all_documents.keys())

bot_vector = tfidf_vectorize(bot_answer, all_words, word_counts_all_documents, total_documents)
manual_vector = tfidf_vectorize(manual_answer, all_words, word_counts_all_documents, total_documents)

similarity = cosine_similarity(bot_vector, manual_vector)
  
  

print(similarity)


