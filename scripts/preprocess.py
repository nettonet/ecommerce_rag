import os
from utils.embeddings import text_to_embedding, chunks_generator
from utils.es_client import ESClient
from configs import settings

def process_documents():
    """处理文本类数据（商品描述、客服对话等）"""
    es = ESClient()
    es.create_index(settings.index_name)
    
    actions = []
    for file in os.listdir(settings.data_path):
        if file.endswith(".txt"):
            with open(os.path.join(settings.data_path, file), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = chunks_generator(text)
                
                for i, chunk in enumerate(chunks):
                    doc_id = f"{file.split('.')[0]}-chunk-{i}"
                    embedding = text_to_embedding(chunk)
                    
                    actions.append({
                        "_index": settings.index_name,
                        "_id": doc_id,
                        "_source": {
                            "text": chunk,
                            "text_vector": embedding.tolist(),
                            "metadata": {"source": file, "chunk_id": i}
                        }
                    })
                    
                    if len(actions) >= settings.batch_size:
                        es.bulk_index(actions)
                        actions.clear()
    
    if actions:
        es.bulk_index(actions)

def process_images():
    """处理图像类数据（商品图片）"""
    es = ESClient()
    for image_file in os.listdir(settings.image_path):
        image_path = os.path.join(settings.image_path, image_file)
        embedding = image_to_embedding(image_path)
        
        es.client.index(
            index=settings.index_name,
            id=image_file,
            document={
                "image_vector": embedding.tolist(),
                "metadata": {"format": image_file.split(".")[-1], "source": "product_images"}
            }
        )

if __name__ == "__main__":
    process_documents()
    process_images()
