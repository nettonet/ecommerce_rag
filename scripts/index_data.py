import os
import json
import numpy as np
from PIL import Image
from elasticsearch.helpers import bulk
from utils.embeddings import text_to_embedding, image_to_embedding, chunks_generator
from utils.es_client import ESClient
from configs import settings  # 导入配置文件

# 初始化Elasticsearch客户端
es_client = ESClient()
es_client.create_index(settings.index_name)  # 创建索引（若不存在）

def index_text_documents():
    """索引文本类数据（商品描述、客服对话等）"""
    print(f"开始索引文本数据，存储路径：{settings.data_path}")
    actions = []
    
    # 遍历所有文本文件
    for filename in os.listdir(settings.data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(settings.data_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if not text:
                    print(f"跳过空文件：{filename}")
                    continue
                
                # 文本分块
                chunks = chunks_generator(text, chunk_size=settings.chunk_size)
                
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = f"{os.path.splitext(filename)[0]}-chunk-{chunk_idx}"
                    text_vector = text_to_embedding(chunk)
                    
                    # 构造Elasticsearch文档
                    actions.append({
                        "_index": settings.index_name,
                        "_id": doc_id,
                        "_source": {
                            "text": chunk,
                            "text_vector": text_vector.tolist(),  # 转换为列表
                            "metadata": {
                                "source": "document",
                                "filename": filename,
                                "chunk_index": chunk_idx
                            }
                        }
                    })
                    
                    # 达到批次大小后批量提交
                    if len(actions) >= settings.batch_size:
                        _ = bulk(es_client.client, actions)  # 忽略返回值
                        print(f"批量提交 {len(actions)} 条文本数据")
                        actions.clear()
    
    # 处理剩余数据
    if actions:
        _ = bulk(es_client.client, actions)
        print(f"批量提交剩余 {len(actions)} 条文本数据")
    print(f"文本数据索引完成，共处理 {len(actions) + len(chunks)} 条数据")

def index_image_documents():
    """索引图像类数据（商品图片）"""
    print(f"开始索引图像数据，存储路径：{settings.image_path}")
    actions = []
    
    # 遍历所有图像文件
    for filename in os.listdir(settings.image_path):
        if filename.lower().endswith(("jpg", "jpeg", "png", "gif")):
            file_path = os.path.join(settings.image_path, filename)
            
            try:
                image_vector = image_to_embedding(file_path)
                
                # 构造Elasticsearch文档（仅存储图像向量和元数据）
                actions.append({
                    "_index": settings.index_name,
                    "_id": filename,  # 使用文件名作为文档ID
                    "_source": {
                        "image_vector": image_vector.tolist(),  # 转换为列表
                        "metadata": {
                            "source": "image",
                            "filename": filename,
                            "format": filename.split(".")[-1],
                            "upload_time": os.path.getmtime(file_path)
                        }
                    }
                })
                
                # 达到批次大小后批量提交
                if len(actions) >= settings.batch_size:
                    _ = bulk(es_client.client, actions)
                    print(f"批量提交 {len(actions)} 条图像数据")
                    actions.clear()
                    
            except Exception as e:
                print(f"跳过损坏的图像 {filename}: {str(e)}")
    
    # 处理剩余数据
    if actions:
        _ = bulk(es_client.client, actions)
        print(f"批量提交剩余 {len(actions)} 条图像数据")
    print(f"图像数据索引完成，共处理 {len(actions)} 条数据")

def main():
    """主函数：执行文本和图像数据的索引"""
    # 清空索引（测试环境可选，生产环境慎用）
    # if es_client.client.indices.exists(settings.index_name):
    #     es_client.client.indices.delete(settings.index_name)
    
    index_text_documents()
    index_image_documents()
    print("所有数据索引完成！")

if __name__ == "__main__":
    main()
