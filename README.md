# finetuning-chatbot

Install packages:

```pip install transformers .datasets accelerate peft trl```
4 fine-tuned model is provided for this repo.
1. guanaco_all_125m -- use model [gpt-neo-125m](https://huggingface.co/EleutherAI/gpt-neo-125m) with [guanaco dataset](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)
2. guanaco_all_1_3b -- use model [gpt-neo-1.3b](https://huggingface.co/EleutherAI/gpt-neo-1.3B) with [guanaco dataset](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)
3. orca_10k_125m -- use [orca dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca)
4. orca-10k_1_3_b -- use [orca dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca)



run loading.ipynb or
```python running.py``` 
