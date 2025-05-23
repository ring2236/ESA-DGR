import json
import threading
import os
from datetime import datetime
from typing import List, Dict

import sys
sys.path.append("/GLOBALFS/sysu_dfli_1/Kexin/MarchTrain/src")
from utils.retrieve import retrieve  # 引入检索模块
from api.model_api import ModelAPI  # 引入模型调用模块

# 全局变量定义
INPUT_FILE = "/GLOBALFS/sysu_dfli_1/Kexin/MarchTrain/baseExp/evidence_SLModel_v0/data/hotpot_dev_distractor_v1_2k.json"
BASE_OUTPUT_DIR = "/GLOBALFS/sysu_dfli_1/Kexin/MarchTrain/baseExp/evidence_SLModel_v0/output/ESLModel"
LOCK = threading.Lock()  # 线程锁
PROCESSED_IDS_FILE = "/GLOBALFS/sysu_dfli_1/Kexin/MarchTrain/baseExp/evidence_SLModel_v0/output/ESLModel/qwen_dev_processed_ids.txt"  # 断点记录文件

# 加载已处理的ID列表
def load_processed_ids() -> set:
    if os.path.exists(PROCESSED_IDS_FILE):
        with open(PROCESSED_IDS_FILE, 'r') as f:
            return set(line.strip() for line in f)
    return set()

# 保存已处理的ID
def save_processed_id(entry_id: str):
    with LOCK:
        with open(PROCESSED_IDS_FILE, 'a') as f:
            f.write(f"{entry_id}\n")

# 生成参考信息
def generate_reference(retrieved_results: List[Dict]) -> str:
    """
    从检索结果中提取 paragraph_text 字段并拼接成参考信息。
    """
    return "\n".join([hit["_source"]["paragraph_text"] for hit in retrieved_results])

# 更新并保存数据到文件
def save_entry(entry: Dict, output_file: str):
    with LOCK:
        # 读取现有数据
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # 更新数据
        data.append(entry)
        
        # 写回文件
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

# 生成 evidence prompt
def generate_evidence_prompt(question: str, reference: str) -> str:
    return json.dumps({"question": question, "evidence": reference})

# 提取 strictModel 的 score 和 missing_evidence
def parse_strict_response(response: str) -> (int, str):
    try:
        parsed_response = json.loads(response)
        score = parsed_response.get("score", 0)
        missing_evidence = parsed_response.get("missing_evidence", "")
        return score, missing_evidence
    except json.JSONDecodeError:
        print("Error parsing strictModel response.")
        return 0, ""

# 提取 looseModel 的 looseScore
def parse_loose_response(response: str) -> float:
    try:
        return float(response.strip())
    except ValueError:
        print("Error parsing looseModel response.")
        return 0.0

# 生成 prompt
def generate_prompt(question: str, reference: str) -> str:
    return f"""
    Based on the following reference information, answer the question in JSON format with "process" and "final answer" fields:
    
    Reference: {reference}
    Question: {question}

    for the final answer,you should follow these two things：
    
    1. If the question is of a yes/no type (e.g., "am", "is", "are"), only answer "yes" or "no".
    2. If the question is about "who", "where", "when", "what", etc., answer with only the relevant information (e.g., name, location, date, etc.), avoiding full sentences.
    
    For example: 
    The output should be concise and without explanation. Only return the most relevant keywords or key information, formatted as a phrase or value. For example:
    - If the question is "Who is the president of the USA?" and the sub-answer mentions "Joe Biden" or "current president", answer only "Joe Biden" or "President Biden".
    - If the question is "Is it raining?" and the sub-answer suggests yes or no, answer "yes" or "no".
    
    Output format:
    {{
        "process": "<your reasoning process here>",
        "final answer": "<your final answer here>"
    }}
    """

def getEvidencePrompt(current_reference_raw: str, question: str) -> str:
    return f"""
        "Extract the current evidence from the provided reference text, ensuring it is concise, relevant, and strictly limited to information directly related to the question. "
        "The output should be a short paragraph of 3-5 sentences, presenting only the key details in a single block without bullet points. "
        "Remain faithful to the original text and avoid adding any interpretations or additional information."
    
    Question: {question}
    current_reference: {current_reference_raw}
    """

# 单条数据处理逻辑
def process_entry(entry: Dict, model_api: ModelAPI, default_model: str, default_role: str,
                  strict_model: str, strict_role: str, loose_model: str, loose_role: str,
                  loose_score_threshold: float, max_round: int, output_file: str):
    entry_id = entry["_id"]
    
    # 检查是否已处理
    if entry_id in processed_ids:
        print(f"Skipping already processed entry: {entry_id}")
        return
    
    # 初始化变量
    round_num = 1
    question = entry["question"]
    cumulative_reference = ""  # 累积的 reference
    round_logs = {}  # 记录每一轮的日志信息

    while round_num <= max_round:
        # 检索相关段落
        retrieved_results = retrieve(query=question if round_num == 1 else missing_evidence, corpus_name="hotpotqa", size=10)
        current_reference_raw = generate_reference(retrieved_results)


        # 注意⚠️！这里加入Evidence Extract 函数，把得到的current_reference_raw经过一层model提炼出来
        EvidencePrompt = getEvidencePrompt(current_reference_raw,question)
        current_reference = model_api.get_response(default_model, default_role, EvidencePrompt)
        cumulative_reference += "\n" + current_reference  # 拼接累积的 reference
        
        # 构造 evidence prompt
        evidence_prompt = generate_evidence_prompt(question, cumulative_reference)
        
        # 调用 strictModel
        try:
            strict_response = model_api.get_response(strict_model, strict_role, evidence_prompt)
            strict_score, missing_evidence = parse_strict_response(strict_response)
        except Exception as e:
            print(f"Error calling strictModel for entry {entry_id}: {e}")
            break

        # 判断 strict_score 是否为 1
        if strict_score == 1:
            # 如果 strict_score 为 1，跳过 looseModel，直接进入 defaultModel 处理
                    # 记录当前轮次的日志
            round_logs[f"round_{round_num}"] = {
                "strict_score": strict_score,
                "missing_evidence": missing_evidence,
                "current_reference": current_reference
            }
            print(f"Strict score is 1 for entry {entry_id}, skipping looseModel.")
            break
        # 调用 looseModel

        try:
            loose_response = model_api.get_response(loose_model, loose_role, evidence_prompt)
            loose_score = parse_loose_response(loose_response)
        except Exception as e:
            print(f"Error calling looseModel for entry {entry_id}: {e}")
            break
        
        # 记录当前轮次的日志
        round_logs[f"round_{round_num}"] = {
            "strict_score": strict_score,
            "missing_evidence": missing_evidence,
            "loose_score": loose_score,
            "current_reference": current_reference,
        }
        
        # 判断退出条件
        if loose_score >= loose_score_threshold:
            break
        else:
            round_num += 1
    
    # 调用 defaultModel
    try:
        prompt = generate_prompt(question, cumulative_reference)  # 使用累积的 reference
        answer = model_api.get_response(default_model, default_role, prompt)
        parsed_answer = json.loads(answer)
        entry['Answer_process'] = parsed_answer.get('process', '')
        entry['Answer_final'] = parsed_answer.get('final answer', '')
    except Exception as e:
        print(f"Error calling defaultModel for entry {entry_id}: {e}")
        entry['Answer_process'] = "Error generating answer"
        entry['Answer_final'] = ""
    
    # 保存结果
    entry['retrieved_passages'] = cumulative_reference.strip()  # 去掉多余的换行符
    entry['round_logs'] = round_logs  # 添加轮次日志信息
    save_entry(entry, output_file)
    save_processed_id(entry_id)
    print(f"Processed entry: {entry_id}")

# 多线程处理主函数
def main(default_model: str, default_role: str, strict_model: str, strict_role: str,
         loose_model: str, loose_role: str, loose_score_threshold: float, max_round: int, num_threads: int = 5):
    global processed_ids
    processed_ids = load_processed_ids()
    
    # 初始化模型 API
    config_path = "/GLOBALFS/sysu_dfli_1/Kexin/MarchTrain/src/config/config.json"
    model_api = ModelAPI(config_path)
    
    # 加载数据
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    # 动态生成输出文件路径
    current_date = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(BASE_OUTPUT_DIR, f"{default_model}_{current_date}.json")
    
    # 确保输出目录存在
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # 创建线程池
    threads = []
    for entry in data:
        while len(threads) >= num_threads:
            # 清理已完成的线程
            threads = [t for t in threads if t.is_alive()]
        
        # 启动新线程
        thread = threading.Thread(target=process_entry, args=(
            entry, model_api, default_model, default_role, strict_model, strict_role,
            loose_model, loose_role, loose_score_threshold, max_round, output_file
        ))
        thread.start()
        threads.append(thread)
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process HotpotQA dataset with retrieval and model inference.")
    parser.add_argument("--defaultModel", type=str, required=True, help="Default model name")
    parser.add_argument("--defaultRole", type=str, required=True, help="Default system role key")
    parser.add_argument("--strictModel", type=str, required=True, help="Strict model name")
    parser.add_argument("--strictRole", type=str, required=True, help="Strict system role key")
    parser.add_argument("--looseModel", type=str, required=True, help="Loose model name")
    parser.add_argument("--looseRole", type=str, required=True, help="Loose system role key")
    parser.add_argument("--looseScore", type=float, required=True, help="Loose score threshold")
    parser.add_argument("--maxRound", type=int, required=True, help="Maximum number of rounds")
    parser.add_argument("--threads", type=int, default=5, help="Number of threads to use")
    args = parser.parse_args()
    
    main(
        args.defaultModel, args.defaultRole, args.strictModel, args.strictRole,
        args.looseModel, args.looseRole, args.looseScore, args.maxRound, args.threads
    )