export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/huyang/hidden_flow
pip install transformers_stream_generator


# export hucfg_use_randn=True
# python hidden_flow_all.py 

# export hucfg_use_randn=False
# python hidden_flow_all.py 


python eval.py >> log.txt
