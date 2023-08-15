from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM
from file_reader import read_file

import argparse
import warnings

warnings.filterwarnings("ignore")
logging.set_verbosity(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--prompt',
    default='Tell me about AI',
    help='the user prompt/question',
    type=str
)
parser.add_argument(
    '--system-agenga',
    '-sa',
    dest='system_agenda',
    help='what should the system do, summarize, answer a question...',
    default="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
)
parser.add_argument(
    '--limit',
    default=11000,
    type=int,
    help='set limit for how many characeters to use from the prompt, \
          -1 indicates to use the whole prompt, ⬆️ character ⬆️ vram needed'
)
args = parser.parse_args()

EXTS = ['.txt', '.pdf']

model_name_or_path = "Llama-2-7b-Chat-GPTQ"
model_basename = "gptq_model-4bit-128g"

use_triton = False
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device=device,
        use_triton=use_triton,
        quantize_config=None)

prompt = args.prompt
# Assuming it may be file ends with `EXTS`.
for ext in EXTS:
    if args.prompt.endswith(ext):
        prompt = read_file(args.prompt)
    
system_message = args.system_agenda
prompt_template=f'''[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt} [/INST]'''

if args.limit != -1:
    prompt_template = prompt_template[:args.limit] + ' [/INST]'
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15,
)

model_outputs = pipe(prompt_template)[0]['generated_text']
outputs = model_outputs.split('[/INST]')[-1]
print(outputs)
