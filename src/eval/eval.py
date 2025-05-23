import sys
import ujson as json
import re
import string
from collections import Counter

def normalize_answer(s):
    """规范化答案以便进行比较"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """计算 F1 分数"""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    """计算精确匹配分数"""
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    """更新答案的评估指标"""
    # 检查 prediction 和 gold 是否为字典类型
    if isinstance(prediction, dict):
        prediction = " ".join(f"{k}: {v}" for k, v in prediction.items())
    if isinstance(gold, dict):
        gold = " ".join(f"{k}: {v}" for k, v in gold.items())

    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def eval(prediction_file, gold_file):
    """主评估函数"""
    # 加载预测文件和真实值文件
    with open(prediction_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # 初始化评估指标
    metrics = {
        'em': 0, 'f1': 0, 'prec': 0, 'recall': 0
    }

    # 遍历真实值文件进行评估
    for gold_id, gold_answer in gold_data.items():
        if gold_id not in predictions:
            print(f'Warning: Missing prediction for ID {gold_id}')
            continue
        predicted_answer = predictions[gold_id]
        update_answer(metrics, predicted_answer, gold_answer)

    # 计算平均值
    num_samples = len(gold_data)
    if num_samples == 0:
        print("No data to evaluate!")
        return

    for key in metrics.keys():
        metrics[key] /= num_samples

    # 输出评估结果
    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval.py <prediction_file> <gold_file>")
        sys.exit(1)

    prediction_file = sys.argv[1]
    gold_file = sys.argv[2]

    eval(prediction_file, gold_file)