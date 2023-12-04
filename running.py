# Load the model
from transformers import AutoModelForCausalLM, AutoTokenizer


class args:
    model_name_or_path = "./guanaco_all_125m"
    cache_dir = "./cache/"
    model_revision = "main"
    use_fast_tokenizer = True


tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    use_fast=args.use_fast_tokenizer,
    revision=args.model_revision,
    use_auth_token=None,
)

pt_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

input_text = "what's machine learning?"

inputs = tokenizer(input_text, return_tensors="pt", padding=True)

output = pt_model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.9,
    top_k=30,
    top_p=0.95,
    repetition_penalty=1.2,
    num_return_sequences=1
)
response_text = tokenizer.decode(output[0], skip_special_tokens=True)


print(response_text)
