import chromadb
import pandas as pd
import traceback

from chromadb.utils import embedding_functions
from datetime import datetime
from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"


def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )

    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    csv_file = "COA_OpenData.csv"
    df = pd.read_csv(csv_file)

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        host_words = str(row.get("HostWords", ""))
        documents.append(host_words)

        create_date = row.get("CreateDate", "")

        try:
            timestamp = int(datetime.strptime(create_date, "%Y-%m-%d").timestamp())
            print(timestamp)
        except (ValueError, TypeError):
            timestamp = 0

        metadata = {
            "file_name": csv_file,
            "name": str(row.get("Name", "")),
            "type": str(row.get("Type", "")),
            "address": str(row.get("Address", "")),
            "tel": str(row.get("Tel", "")),
            "city": str(row.get("City", "")),
            "town": str(row.get("Town", "")),
            "date": timestamp
        }

        print(metadata)
        metadatas.append(metadata)
        print(f"record_{index}")
        ids.append(f"record_{index}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return collection


def generate_hw02(question, city, store_type, start_date, end_date):
    pass


def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass


def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    print(collection)
    return collection


def main():
    # demo(None)
    generate_hw01()


if __name__ == '__main__':
    main()
