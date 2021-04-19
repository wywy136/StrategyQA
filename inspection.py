import json
import random


file = open("./data/strategyqa_train.json", 'r', encoding='utf-8')
json_data: list = json.load(file)
num = len(json_data)
dev_num = 0.1 * len(json_data)
test_num = 0.1 * len(json_data)
train_num= 0.8 * len(json_data)
train_data = []
dev_data = []
test_data = []
train_file = open("./data/strategyqa_train_train.json", 'w', encoding='utf-8')
test_file = open("./data/strategyqa_train_test.json", 'w', encoding='utf-8')
dev_file = open("./data/strategyqa_train_dev.json", 'w', encoding='utf-8')

while len(json_data) > 0:
    i = random.randint(0, len(json_data)-1)
    data = json_data[i]
    if len(json_data) > test_num + dev_num:
        train_data.append(data)
    elif len(json_data) > test_num:
        dev_data.append(data)
    else:
        test_data.append(data)
    json_data.remove(data)

json.dump(train_data, train_file, indent=4)
json.dump(test_data, test_file, indent=4)
json.dump(dev_data, dev_file, indent=4)