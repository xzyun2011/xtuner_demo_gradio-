import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
 
 
base_path = './demo_model'
os.system(f'git clone https://code.openxlab.org.cn/xzyun2011/xtuner_demo.git ./')
os.system(f'cd {base_path} && git lfs pull')
 
tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()
 
def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response
 
gr.ChatInterface(chat,
                 title="InternLM2-Chat-1.8B",
                description="""
InternLM2-Chat-1.8B is mainly developed by jin.  
                 """,
                 ).queue(1).launch()
