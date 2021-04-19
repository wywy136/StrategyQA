from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
s1 = "Hello Chicago."
print(type(tokenizer(s1)["input_ids"]))