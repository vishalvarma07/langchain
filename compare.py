from openai import OpenAI
import os
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI()
# Set your OpenAI API key

# Function to create embeddings for a sentence using Ada-002
def create_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# Function to compute cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Example sentences
sentence1 = "The banish 30 is .30 caliber silencer"
sentence2 = "How do i buy"

# Create embeddings for each sentence
embedding1 = create_embedding(sentence1)
embedding2 = create_embedding(sentence2)

# print(embedding1.data[0].embedding)

# Compute cosine similarity between the two embeddings
similarity = cosine_similarity(embedding1, embedding2)

print(f"Cosine Similarity between the sentences: {similarity:.4f}")
