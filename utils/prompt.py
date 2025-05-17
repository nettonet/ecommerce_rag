from typing import List, Dict, Optional
import json
import re
from configs import settings  # 导入配置文件

class PromptBuilder:
    def __init__(self, system_prompt: Optional[str] = None):
        """
        初始化提示词构建器
        
        Args:
            system_prompt: 系统级提示词（如角色设定、能力说明）
        """
        self.system_prompt = system_prompt or settings.DEFAULT_SYSTEM_PROMPT  # 可从配置文件加载默认值
        self.conversation_history = []  # 存储对话历史
        self.retrieved_contexts = []    # 存储检索结果
        
    def reset(self):
        """重置对话状态"""
        self.conversation_history.clear()
        self.retrieved_contexts.clear()
        
    def add_conversation(self, role: str, content: str):
        """
        添加对话记录
        
        Args:
            role: 角色（user/assistant）
            content: 对话内容
        """
        self.conversation_history.append({"role": role, "content": content})
        
    def add_retrieved_context(self, contexts: List[str]):
        """
        添加检索结果
        
        Args:
            contexts: 检索到的上下文列表
        """
        self.retrieved_contexts.extend(contexts)
        
    def build_rag_prompt(
        self, 
        user_query: str, 
        max_contexts: int = 5, 
        format_requirement: str = "请用JSON格式返回商品信息"
    ) -> str:
        """
        构建RAG提示词（含系统提示、对话历史、检索结果、格式要求）
        
        Args:
            user_query: 用户原始问题
            max_contexts: 最多包含的检索结果数量
            format_requirement: 输出格式要求（如JSON/Markdown）
            
        Returns:
            完整提示词
        """
        # 截断检索结果数量
        contexts = self.retrieved_contexts[:max_contexts]
        
        # 构建提示词结构
        prompt_parts = [
            f"系统提示：{self.system_prompt}",
            "\n".join([f"历史对话：{item['content']}" for item in self.conversation_history]),
            f"\n用户当前问题：{user_query}",
            f"\n检索到的相关信息（共{len(contexts)}条）：\n" + "\n".join([f"- {ctx}" for ctx in contexts]),
            f"\n格式要求：{format_requirement}",
            "回答要求：必须基于检索信息，若信息不足请说明需要补充的内容"
        ]
        
        return "\n\n".join(prompt_parts).strip()
    
    def build_function_call_prompt(
        self, 
        function_name: str, 
        parameters: Dict[str, str]
    ) -> str:
        """
        构建函数调用提示词（适用于工具调用场景）
        
        Args:
            function_name: 函数名
            parameters: 函数参数
            
        Returns:
            带函数调用格式的提示词
        """
        return f"""请调用工具函数获取信息：
        {{
            "name": "{function_name}",
            "parameters": {json.dumps(parameters)}
        }}"""
    
    def extract_function_call(self, response: str) -> Optional[Dict]:
        """
        从模型响应中提取函数调用参数
        
        Args:
            response: 模型返回的文本
            
        Returns:
            解析后的函数调用参数（JSON格式）
        """
        pattern = r'\{.*?\}'  # 匹配JSON格式内容
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None
    
    def format_answer(
        self, 
        answer: str, 
        source_references: List[str] = None
    ) -> str:
        """
        格式化最终回答（添加来源引用）
        
        Args:
            answer: 回答内容
            source_references: 来源索引列表
            
        Returns:
            格式化后的回答
        """
        if source_references:
            return f"{answer}\n\n引用来源：{', '.join(source_references)}"
        return answer

# 示例用法
if __name__ == "__main__":
    # 初始化构建器
    builder = PromptBuilder(
        system_prompt="你是电商客服助手，负责回答商品相关问题，需优先使用检索到的信息"
    )
    
    # 添加对话历史
    builder.add_conversation("user", "你好，我想买一件羽绒服")
    builder.add_conversation("assistant", "请问你需要什么类型的羽绒服？比如长款/短款、填充物类型等")
    
    # 添加检索结果
    builder.add_retrieved_context([
        "商品A：长款羽绒服，白鸭绒填充，含绒量90%",
        "商品B：短款羽绒服，灰鹅绒填充，含绒量85%"
    ])
    
    # 构建RAG提示词
    prompt = builder.build_rag_prompt(
        user_query="长款羽绒服的填充物是什么？",
        format_requirement="请用中文简洁回答"
    )
    print("生成的提示词：\n", prompt)
    
    # 模拟模型回答
    model_response = "商品A的填充物是白鸭绒，含绒量90%。"
    formatted_answer = builder.format_answer(model_response, source_references=["商品A"])
    print("\n格式化后的回答：\n", formatted_answer)
