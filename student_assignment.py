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
    # 初始化 ChromaDB 客戶端並獲取 Collection
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

    # 第一步：更新指定店家的 new_store_name
    # 查找匹配 store_name 的記錄
    existing_data = collection.get(where={"name": store_name}, include=["metadatas"])

    if existing_data["ids"]:  # 如果找到匹配的店家
        for idx in existing_data["ids"]:
            metadata = collection.get(ids=[idx], include=["metadatas"])["metadatas"][0]
            # 更新元數據，添加 new_store_name
            metadata["new_store_name"] = new_store_name
            collection.update(
                ids=[idx],
                metadatas=[metadata]
            )

    # 第二步：根據查詢條件查詢店家
    where_conditions = []
    if city and isinstance(city, list) and city:
        where_conditions.append({"city": {"$in": city}})
    if store_type and isinstance(store_type, list) and store_type:
        where_conditions.append({"type": {"$in": store_type}})

    where = {"$and": where_conditions} if where_conditions else None

    results = collection.query(
        query_texts=[question],
        n_results=10,  # 返回最多 10 個結果
        where=where,
        include=["metadatas", "distances"]
    )

    # 處理查詢結果
    store_names = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        similarity = 1 - distance
        if similarity >= 0.80:  # 只保留相似度 >= 0.80 的結果
            # 若有 new_store_name，則使用它；否則使用原始 name
            display_name = metadata.get("new_store_name", metadata["name"])
            store_names.append((display_name, similarity))

    # 按相似度分數遞減排序
    store_names.sort(key=lambda x: x[1], reverse=True)

    # 返回僅店家名稱的列表
    return [name for name, _ in store_names]


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

    # question = "我想要找有關茶餐點的店家"
    # city = ["宜蘭縣", "新北市"]
    # store_type = ["美食"]
    # start_date = datetime(2024, 4, 1)
    # end_date = datetime(2024, 5, 1)
    # print(generate_hw02(question, city, store_type, start_date, end_date))

    question = "我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵"
    store_name = "耄饕客棧"
    new_store_name = "田媽媽（耄饕客棧）"
    city = ["南投縣"]
    store_type = ["美食"]
    print(generate_hw03(question, store_name, new_store_name, city, store_type))


if __name__ == '__main__':
    main()
