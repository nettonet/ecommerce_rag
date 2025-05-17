from elasticsearch import Elasticsearch
from configs import settings

class ESClient:
    def __init__(self):
        self.client = Elasticsearch(
            hosts=settings.es_hosts,
            basic_auth=(settings.es_username, settings.es_password),
            ca_certs=settings.es_ca_certs,
            verify_certs=True
        )
    
    def create_index(self, index_name: str):
        """创建混合索引（文本+向量）"""
        mapping = {
            "mappings": {
                "properties": {
                    "text": {"type": "text", "analyzer": "standard"},
                    "text_vector": {
                        "type": "dense_vector",
                        "dims": 384,  # Sentence-BERT输出维度
                        "similarity": "cosine"
                    },
                    "image_vector": {
                        "type": "dense_vector",
                        "dims": 512,  # CLIP输出维度
                        "similarity": "cosine"
                    },
                    "metadata": {"type": "object"}
                }
            }
        }
        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body=mapping)
    
    def bulk_index(self, actions: list[dict]):
        """批量索引数据"""
        from elasticsearch.helpers import bulk
        bulk(self.client, actions)
    
    def hybrid_search(self, query: str, image_path: str = None, top_k: int = 10):
        """混合搜索（BM25+向量搜索）"""
        query_body = {
            "query": {
                "bool": {
                    "should": [
                        # BM25文本搜索
                        {"match": {"text": query}},
                        # 文本向量搜索
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.text_vector, 'text_vector') + 1.0",
                                    "params": {"text_vector": text_to_embedding(query)}
                                }
                            }
                        }
                    ]
                }
            },
            "size": top_k
        }
        
        # 若有图像查询，添加图像向量搜索
        if image_path:
            image_vector = image_to_embedding(image_path)
            query_body["query"]["bool"]["should"].append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.image_vector, 'image_vector') + 1.0",
                        "params": {"image_vector": image_vector.tolist()}
                    }
                }
            })
        
        return self.client.search(index=settings.index_name, body=query_body)
