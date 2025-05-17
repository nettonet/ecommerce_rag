from flask import Flask, request, jsonify
from utils.es_client import ESClient
from transformers import pipeline
import configs.settings as settings

app = Flask(__name__)
es = ESClient()
generator = pipeline("text-generation", model="TheBloke/Llama-2-7B-Chat-GPTQ")  # 示例生成模型

def build_prompt(query: str, context: list[str]) -> str:
    """构建提示词"""
    prompt = f"用户问题：{query}\n检索结果：\n" + "\n".join([f"- {c}" for c in context])
    prompt += "\n请根据以上信息回答用户问题，若信息不足请说明需要更多细节。"
    return prompt

@app.route("/chat", methods=["POST"])
def handle_chat():
    data = request.json
    query = data.get("query")
    image = data.get("image")  # 可选的图像文件路径
    
    # 执行混合搜索
    search_results = es.hybrid_search(query, image_path=image)
    contexts = [hit["_source"]["text"] for hit in search_results["hits"]["hits"]]
    
    # 生成回答
    prompt = build_prompt(query, contexts)
    response = generator(prompt, max_length=512)[0]["generated_text"]
    
    return jsonify({"answer": response, "sources": contexts})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=True)