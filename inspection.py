import json


file = open("./data/StrategyQA/strategyqa_train_paragraphs.json", 'r', encoding='utf-8')
json_data = json.load(file)

print(json_data["17th century-1"])