from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
# Your OpenAI API key
## key is automatically imported from OPENAI_API_KEY when this is used
client = OpenAI()
# api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = api_key

# Function to read text from a file
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to create embeddings for a text using Ada-002
def create_embeddings(text, model = "text-embedding-3-large"):
    response = client.embeddings.create(
        temperature=0.2,
        model=model,
        input=text
    )
    return response['embedding']

# Directory where your text files are stored
folder_path = 'path/to/your/folder'

# List all files in the directory
files = os.listdir(folder_path)

# Process each file and create embeddings
for file_name in files:
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)
        text = read_file(file_path)
        embedding = create_embeddings(text)
        print(f"Embedding created for file: {file_name}")
        # Optionally, you can save the embeddings or further process them
