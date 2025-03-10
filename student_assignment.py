import chromadb
import pandas as pd
import re
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

    metadatas = []
    ids = []

    city_pattern = re.compile(r"^(.*?[市縣])")
    town_pattern = re.compile(r"^(.*?[市縣].*?[區鄉市])")
    town_pattern2 = re.compile(r"^(.*?[市縣].*?[鎮])")

    documents = df["HostWords"].tolist()

    for index, row in df.iterrows():
        create_date = row["CreateDate"]

        # try:
        timestamp = int(datetime.strptime(create_date, "%Y-%m-%d").timestamp())
            # print(timestamp)
        # except (ValueError, TypeError):
        #     timestamp = 0

        # city_csv = str(row.get("City", ""))
        # town_csv = str(row.get("Town", ""))
        # address = row["Address"]
        # city, town = parse_city_town(address)
        #
        # if city_csv != city:
        #     print("city_csv: " + city_csv)
        #     print("city: " + city)
        #
        # if town_csv != town:
        #     print("town_csv: " + town_csv)
        #     print("town: " + town)

        # if not is_valid_city(city) or not is_valid_town(town):
        #     print(city)
        #     print(town)
        #     city_from_address, town_from_address = parse_city_town(address)
        #     city = city_from_address if not is_valid_city(city) else city
        #     town = town_from_address if not is_valid_town(town) else town

        city = re.match(city_pattern, row["Address"]).group()

        if re.match(town_pattern, row["Address"]) is not None:
            town = re.match(town_pattern, row["Address"]).group().split(city)[1]
        elif re.match(town_pattern2, row["Address"]) is not None:
            town = re.match(town_pattern2, row["Address"]).group().split(city)[1]
        else:
            town = city
            city = city.replace("市", "縣")
            print(town)
            print(city)

        metadata = {
            "file_name": csv_file,
            "name": row["Name"],
            "type": row["Type"],
            "address": row["Address"],
            "tel": row["Tel"],
            "city": city,
            "town": town,
            "date": timestamp
        }

        # print(metadata)
        metadatas.append(metadata)
        print(str(index))
        ids.append(str(index))

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
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


def is_valid_city(city):
    return bool(city and isinstance(city, str) and ("縣" in city or "市" in city))


def is_valid_town(town):
    return bool(town and isinstance(town, str) and any(x in town for x in ["鄉", "鎮", "區", "市"]))


def parse_city_town(address):
    if not address or pd.isna(address):
        return "", ""

    first_address = address.split('/')[0].strip()

    city_pattern = r"([^縣市]+[縣市])"
    town_pattern = r"[縣市]([^鄉鎮區市]+[鄉鎮區市])"

    city_match = re.search(city_pattern, first_address)
    town_match = re.search(town_pattern, first_address)

    city = city_match.group(0) if city_match else ""
    town = town_match.group(1) if town_match else ""

    return city, town


def main():
    # demo(None)
    generate_hw01()


if __name__ == '__main__':
    main()
