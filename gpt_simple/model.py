
from gptconfig import GPT1Config
from gpt import GPT
vocab_size = 10
max_len = 12

config = GPT1Config(vocab_size, max_len)
model = GPT(config)