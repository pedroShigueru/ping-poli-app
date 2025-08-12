from sentence_transformers import SentenceTransformer
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')
mongo_client = MongoClient(MONGO_URI)

def connection_mongodb():
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["pingpoli"]
    collection = db["members_informations"]
    
    return collection

def insert_document_in_mongodb(file_name: str, text: str, embedding, collection):
    document = {
            'file_name': file_name,
            'text': text,
            'embedding': embedding
        }

    collection.insert_one(document)

def transform_sentence_to_embedding(sentence: str) -> list:
    model = SentenceTransformer('PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir')
    embedding = model.encode(sentence)
    
    return embedding

directory = "./data/raw"
collection = connection_mongodb()
result = collection.delete_many({})

for file in os.listdir(directory):
    full_path = os.path.join(directory, file)

    if os.path.isfile(full_path):
        with open(full_path, "r", encoding="utf-8") as f:

            member_text = f.read()
            embedding = transform_sentence_to_embedding(member_text).tolist()

            insert_document_in_mongodb(file, member_text, embedding, collection)