import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from configs import settings

# 初始化文本嵌入模型
text_model = SentenceTransformer(settings.text_embedding_model)

# 初始化图像嵌入模型
image_processor = CLIPProcessor.from_pretrained(settings.image_embedding_model)
image_model = CLIPModel.from_pretrained(settings.image_embedding_model).to("cuda")

def text_to_embedding(text: str) -> np.ndarray:
    """生成文本向量"""
    return text_model.encode(text, normalize_embeddings=True)

def image_to_embedding(image_path: str) -> np.ndarray:
    """生成图像向量"""
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        features = image_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().squeeze()

def chunks_generator(text: str, chunk_size: int = 512) -> list[str]:
    """文本分块生成器"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - settings.chunk_overlap)]
