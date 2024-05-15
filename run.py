import argparse
import torch

from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import pipeline

argp = argparse.ArgumentParser()
argp.add_argument('model_dir', help="model directory")
argp.add_argument('--bits', default=0, type=int)
argp.add_argument('--template', default="", type=str)
argp.add_argument('--temperature', default=0.9, type=float)
argp.add_argument('--top_p', default=1.0, type=float)
argp.add_argument('--max_tokens', default=20, type=int)
args = argp.parse_args()

assert args.bits == 4 or args.bits == 8, \
    "quantization should either be 4 or 8 bits"

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

device = "cuda" if torch.cuda.is_available() and args.bits == 0 else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = LlamaForCausalLM.from_pretrained(args.model_dir, 
                                         load_in_4bit=args.bits == 4,
                                         load_in_8bit=args.bits == 8,
                                         torch_dtype=torch.float16)
model = model.to(device)
model.eval()

config = GenerationConfig(
    do_sample=True,
    temperature=args.temperature,
    max_new_tokens=args.max_tokens,
    top_p=args.top_p,
)

task = "text-generation"
pipe = pipeline(task, model=model, tokenizer=tokenizer, device=device)

if args.template.casefold() == "ALPACA".casefold():
    print("Please type the instruction:")
    instruction = input()
    print("Please type the input:")
    input = input()
    
    prompt = ALPACA_TEMPLATE.format(
        instruction=instruction,
        input=input,
    )
else:
    print("Please type the input:")
    prompt = input()

output = pipe(prompt)[0]['generated_text'].split(prompt, 1)[1]

print("Output:\n", output)
