# StrategyQA

### 代码结构

- data/：数据
- dataset/：Dataset类
- golden_sentence_predictor/：预测关键句的model
- model/：Reasoning模型
- trainer/：训练
- config.py：配置
- evaluator.py：评价
- main.py
- 其他文件没用

### 数据

数据都在data/路径下，其中strategyqa/路径下存放的4个文件是官网提供的训练数据和测试数据，其他json文件分别为

- dev：论文作者从train中划分出来的验证集
- train：论文作者从train中划分出来的小train
- dev_gdsent：包含预测出的golden sentence的dev
- train_gdsent：包含预测出的golden sentenced的小train
- transformer_qa_ORA-P_dev_no_placeholders：将dev中的占位符都替换为预测结果，复现last-step那个模型用
- transformer_qa_ORA-P_train_no_placeholders：同上

### 论文里的model

论文的github链接：[eladsegal/strategyqa: The official code of TACL 2021, "Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies". (github.com)](https://github.com/eladsegal/strategyqa)

其中提供了很多checkpoint，包括用BART做分解的那个model