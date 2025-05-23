import json
import os
import sys
import requests
from openai import OpenAI

# 保留旧有本地模块搜索路径
sys.path.append("/GLOBALFS/sysu_dfli_1/Kexin/MarchTrain/src")

class ModelAPI:
    def __init__(self, config_path):
        # 1. 加载主配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
       
        # 2. 准备容器
        self.clients = {}             # 存放 OpenAI 客户端实例
        self.model_types = {}         # 标记模型调用类型：local / external / requests
        self.model_features = {}      # 标记模型特殊能力（如 reasoning）
        self.external_base_urls = {}  # 存放外部模型的 base_url，requests 调用时使用
        
        # 3. 处理本地模型（config['models'] 中的条目）
        for model_name, base_url in self.config.get('models', {}).items():
            self.clients[model_name] = OpenAI(api_key='EMPTY', base_url=base_url)
            self.model_types[model_name] = 'local'
        
        # 4. 处理 external.json 中的外部 API 模型
        external_config_path = "/GLOBALFS/sysu_dfli_1/Kexin/MarchTrain/src/config/external.json"
        if os.path.exists(external_config_path):
            with open(external_config_path, 'r') as f:
                external_config = json.load(f)
            
            for model_name, api_conf in external_config.get('models', {}).items():
                api_key  = api_conf.get('api_key')
                base_url = api_conf.get('base_url')
                real_name = api_conf.get('model_name')
                
                # 用 OpenAI 客户端保存认证信息（主要为了统一管理 api_key）
                self.clients[model_name] = OpenAI(api_key=api_key, base_url=base_url)
                
                # 根据 URL 判断调用方式：SiliconFlow 用 requests，其它用 openai-style
                if "siliconflow.cn" in base_url:
                    self.model_types[model_name] = 'requests'
                else:
                    self.model_types[model_name] = 'external'
                
                # 记录 base_url 以便 requests.post 时使用
                self.external_base_urls[model_name] = base_url
                
                # 记录实际的模型名称
                self.config.setdefault('external_model_names', {})[model_name] = real_name
                
                # deepseek-reasoner 有推理功能
                if real_name == 'deepseek-r1-250120':
                    self.model_features[model_name] = {'has_reasoning': True}
        
        # 5. 加载 evidence 提示内容
        with open("/GLOBALFS/sysu_dfli_1/Kexin/MarchTrain/src/prompts/evidence.json", 'r') as f:
            self.evidence_prompts = json.load(f)
    
    def get_response(self, model_name, system_role_key, query):
        # 验证模型和角色键
        if model_name not in self.clients:
            raise ValueError(f"Model '{model_name}' is not available.")
        if system_role_key not in self.evidence_prompts:
            raise ValueError(f"System role key '{system_role_key}' is not defined.")
        
        # 构建 messages
        has_reason = self.model_features.get(model_name, {}).get('has_reasoning', False)
        system_content = self.evidence_prompts[system_role_key]
        if has_reason:
            messages = [{'role': 'user', 'content': query}]
        else:
            messages = [
                {'role': 'system', 'content': system_content},
                {'role': 'user',   'content': query}
            ]
        
        mtype = self.model_types.get(model_name)
        client = self.clients[model_name]
        
        # —— external (OpenAI-style) 调用 —— 
        if mtype == 'external':
            actual = self.config['external_model_names'][model_name]
            resp = client.chat.completions.create(
                model=actual,
                messages=messages,
                max_tokens=512,
                temperature=0,
                stream=False
            )
            # print("DEBUG from ModelAPI: the full response is: ",resp)
            return resp.choices[0].message.content
        
        # —— requests (SiliconFlow 风格) 调用 —— 
        elif mtype == 'requests':
            url = self.external_base_urls[model_name]
            payload = {
                "model": self.config['external_model_names'][model_name],
                "messages": messages,
                "stream": False,
                "max_tokens": 512,
                "stop": None,
                "temperature": 0.7,   # 可根据需要自行调整
                "top_p": 0.7,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "n": 1,
                "response_format": {"type": "text"},
                # 如果需要 tools 参数，这里也可加上
            }
            headers = {
                "Authorization": f"Bearer {client.api_key}",
                "Content-Type": "application/json"
            }
            r = requests.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            # 假设返回结构同 OpenAI：choices → message → content
            return data['choices'][0]['message']['content']
        
        # —— local (内部服务) 调用 —— 
        else:  # local
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                temperature=0
            )
            print("DEBUG from ModelAPI: the full response is: ",resp)
            return resp.choices[0].message.content
    
    def get_response_with_reasoning(self, model_name, system_role_key, query):
        # 验证模型和角色键
        if model_name not in self.clients:
            raise ValueError(f"Model '{model_name}' is not available.")
        if system_role_key not in self.evidence_prompts:
            raise ValueError(f"System role key '{system_role_key}' is not defined.")
        
        # 构建 messages（同上）
        has_reason = self.model_features.get(model_name, {}).get('has_reasoning', False)
        system_content = self.evidence_prompts[system_role_key]
        if has_reason:
            messages = [{'role': 'user', 'content': query}]
        else:
            messages = [
                {'role': 'system', 'content': system_content},
                {'role': 'user',   'content': query}
            ]
        
        mtype = self.model_types.get(model_name)
        client = self.clients[model_name]
        result = {}
        
        # —— external —— 
        if mtype == 'external':
            actual = self.config['external_model_names'][model_name]
            resp = client.chat.completions.create(
                model=actual,
                messages=messages,
                max_tokens=512,
                temperature=0,
                stream=False
            )
            print("DEBUG from ModelAPI: the full response is: ",resp)
            result['content'] = resp.choices[0].message.content
        
        # —— requests —— 
        elif mtype == 'requests':
            url = self.external_base_urls[model_name]
            payload = {
                "model": self.config['external_model_names'][model_name],
                "messages": messages,
                "stream": False,
                "max_tokens": 512,
                "stop": None,
                "temperature": 0.7,
                "top_p": 0.7,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "n": 1,
                "response_format": {"type": "text"},
            }
            headers = {
                "Authorization": f"Bearer {client.api_key}",
                "Content-Type": "application/json"
            }
            r = requests.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            result['content'] = data['choices'][0]['message']['content']
        
        # —— local —— 
        else:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                temperature=0
            )
            print("DEBUG from ModelAPI: the full response is: ",resp)
            result['content'] = resp.choices[0].message.content
        
        # 如果支持 reasoning，则尝试提取
        if has_reason:
            # OpenAI client 返回
            if mtype != 'requests' and hasattr(resp.choices[0].message, 'reasoning_content'):
                result['reasoning_content'] = resp.choices[0].message.reasoning_content
            else:
                # SiliconFlow 或未返回时的 fallback
                result['reasoning_content'] = "模型支持推理功能，但未返回相关内容。"
        
        return result
