import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
    revision="v2.0",
    use_fast=False,
    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
    revision="v2.0",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", revision="v2.0")
messages = []
j = {"task": "RE", "source": "CMeIE", "instruction": "{\"instruction\": \"你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。\", \"schema\": [\"筛查\", \"并发症\", \"相关（导致）\", \"病理生理\"], \"input\": \"稳定型缺血性心脏疾病@无创负荷试验用于明确缺血性心脏病诊断时应局限于依据年龄、性别和症状有中度可能性的患者。\"}", "output": "{\"筛查\": [], \"并发症\": [], \"相关（导致）\": [], \"病理生理\": []}"}

messages.append({"role": "user", "content": j['instruction']})
response = model.chat(tokenizer, messages)
print(response)
import IPython; IPython.embed()