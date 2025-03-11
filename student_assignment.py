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
        api_type=gpt_emb_config['openai_type'],
        api_base=gpt_emb_config['api_base'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )

    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    if collection.count() != 0:
        return collection

    csv_file = "COA_OpenData.csv"
    df = pd.read_csv(csv_file)

    metadatas = []
    ids = []

    city_pattern = re.compile(r"^(.*?[市縣])")
    town_pattern = re.compile(r"^(.*?[市縣].*?[區鄉市])")
    town_pattern2 = re.compile(r"^(.*?[市縣].*?[鎮])")

    documents = df["HostWords"].tolist()

    for index, row in df.iterrows():
        timestamp = int(datetime.strptime(row["CreateDate"], "%Y-%m-%d").timestamp())

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
    client = chromadb.PersistentClient(path="./")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_type=gpt_emb_config['openai_type'],
        api_base=gpt_emb_config['api_base'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )

    collection = client.get_or_create_collection(
        name="TRAVEL",
        embedding_function=openai_ef
    )

    # 構建查詢篩選條件
    where_conditions = []

    # 城市篩選
    if city and isinstance(city, list) and city:
        where_conditions.append({"city": {"$in": city}})

    # 店家類型篩選
    if store_type and isinstance(store_type, list) and store_type:
        where_conditions.append({"type": {"$in": store_type}})

    # 日期範圍篩選（拆分為獨立的 $gte 和 $lte 條件）
    if start_date:
        where_conditions.append({"date": {"$gte": int(start_date.timestamp())}})
    if end_date:
        where_conditions.append({"date": {"$lte": int(end_date.timestamp())}})

    # 使用 $and 組合所有條件
    where = {"$and": where_conditions} if where_conditions else None

    # 執行查詢
    results = collection.query(
        query_texts=[question],
        n_results=10,  # 返回最多 10 個結果
        where=where,
        include=["metadatas", "distances"]  # 返回元數據和距離
    )

    # 處理查詢結果
    store_names = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        # 餘弦相似度轉換：similarity = 1 - distance
        similarity = 1 - distance
        print(metadata)
        print(similarity)
        if similarity >= 0.80:  # 只保留相似度 >= 0.80 的結果
            store_name = metadata["name"]
            store_names.append((store_name, similarity))

    # 按相似度分數遞減排序
    store_names.sort(key=lambda x: x[1], reverse=True)

    # 返回僅店家名稱的列表
    return [name for name, _ in store_names]


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
    # generate_hw01()

    question = "我想要找有關茶餐點的店家"
    city = ["宜蘭縣", "新北市"]
    store_type = ["美食"]
    start_date = datetime(2024, 4, 1)
    end_date = datetime(2024, 5, 1)
    print(generate_hw02(question, city, store_type, start_date, end_date))


if __name__ == '__main__':
    main()
