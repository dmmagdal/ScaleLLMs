# QLora Falcon 7B

Description: This is a quick example of finetuning the Falcon 7B model with QLora given a set of text.


### Setup

 - It is recommended that you create a python virtual environment instead of a conda due to version issues with a lot of necessary packages.
 - To set up the virtual environment, install the `venv` package:
	 - `pip3 install virtualenv`
 - Create the new virtual environment:
	 - `python -m venv autogptq-env`
 - Activate the virtual environment:
	 - Linux/MacOS: `source autogptq-env/bin/activate`
	 - Windows: `.\autogptq-env\Scripts\activate`
 - Deactivate the virtual environment:
	 - `deactivate`
 - Install the necessary packages (while the virtual environment is active):
	 - `(autogptq-env) pip3 install -r requirements.txt`
 - Also be sure to install the necessary version of `pytorch` according to your OS (refer to the `pytorch` website but the following command will help):
	 - Linux & Windows (CUDA 11.8): `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`


### Notes

 - LLM models I want to quantize:
	 - Falcon 1.3B and/or 7B (maybe do 40B if I have the resources)
	 - Llama 2 7B and/or 13B (most likely dont have enough resources for 70B)
	 - T5 (an older model whose uses I have yet to understand but has great zero shot versatility from what I've heard)
	 - Flan-T5 (let's see how the ChatGPT killer really performs)
 - As of writing this (10/07/2023), I have been able to download the `bitsandbytes` module required for quantizing models. However, upon trying to run the programs, I get an error from `bitsandbytes` regarding the module not being able to locate the CUDA version. Going to GitHub, I've seen this is an issue with other users and the repo owner has marked such issues as `[CUDA_SETUP]` and `[low-priority]`. I have low expectations as to whether this will be updated anytime soon.
	 - I am considering alternatives such as GGML/GGUL and GPTQ for quantizing the models. Actual finetuning may be different as a result but I have to get to that part next.
	 - UPDATE 10/10/2023: Using python's `virtualenv` allowed me to get around this issue. I am now able to run `bitsandbytes` to peform the model quantization. I still need to figure out the finetuning part with `peft`. 
 - Testing
	 - Linux (my Darkstar GPU server)
		 - Using `device_map="auto"` works as well as `device_map={"": 0}`.
		 - Be aware that the `torch` version in `requirements.txt` may not work properly on the first try when you set up the `virtualenv`. I'd advise everyone to use the necessary install command from the PyTorch website (just specify the correct version).
 - Quantization
	 - Falcon 7B
		 - Programs
			 - `bitsandbytes_quantize_falcon.py`
			 - `finetune_falcon.py`
		 - Results/Notes
			 - Was able to quantize falcon 7b from `tiiuae/falcon-7b` on my Darkstar GPU server with the `finetune_falcon.py` and `bitsandbytes_quantize_falcon.py` scripts. 
			 - Note that as of now (10/30/2023), saving 4-bit quantized models is not possible. Quantized models as small as 8-bit can be serialized and saved. This extends to both saving locally and pushing to huggingface hub. As a result, the optional parameter `--quantize_4_bit` is available for the `bitsandbytes_quantize_falcon.py` script to determine whether to quantize the model in 8-bit (default) or 4-bit (wont save the model). Since `finetune_falcon.py` only does 4-bit quantization, it has to pull from huggingface hub/cache every time it runs (its only saved outputs are the Lora adapater model outputs from the finetuning).
			 - For the `bitsandbytes_quantize_falcon.py`` script (8-bit quantization), the VRAM usage was around 8.2GB VRAM (so a 12 to 16GB card is very much recommended for 7B models). Actual RAM usage was around 14.5GB but would still recommend using 16GB RAM. The quantization process (running the script) takes around 3 minutes.
			 - The resulting 8-bit quantized model in `bitsandbytes_quantized_8bit_falcon-7b/` folder (and the tokenizer) comes out to around 6.8 GB on disk.
 - Finetuning
	 - Falcon 7B
		 - Programs
			 - `bitsandbytes_finetune_falcon.py`
			 - `finetune_falcon.py`
		 - Results/Notes
			 - Was able to successfully run both scripts to completion.
			 - `bitsandbytes_finetune_falcon.py` uses an 8-bit quantized version of the model that was created and saved locally by `bitsandbytes_quantize_falcon.py` while `finetune_falcon.py` uses a 4-bit quantized version of the model it creats itself (it cannot save it because that is not yet supported).
			- For the `bitsandbytes_finetune_falcon.py`` script (8-bit quantization), the VRAM usage was around 26.1GB VRAM (so a 36 to 48GB card is very much recommended for 7B models; it may be prudent to acquire a few cards for this process actually). Actual RAM usage was around 7.4GB but would still recommend using 16GB RAM. The quantization process (running the script) can take a long time depnending on the number of steps/epochs (100 steps took around 20 minutes).
			 - The resulting adapter model in `results-model/` (or `8bit-falcon7b-results-model/` if using `bitsandbytes_finetune_falcon.py`) folder (and the tokenizer) comes out to around 73 MB on disk.
 - Exporting to ONNX
	 - Created and ran the `export_onnx_falcon.py` script to test if I can export falcon 7b (4-bit quantized with `bitsandbytes`) to ONNX based one the code found on this [blog post](https://huggingface.co/blog/convert-transformers-to-onnx). The script takes only the quantized model into consideration, **not** a quantized model that's been finetuned with `peft`.
	 - 10/30/2023: Ran the first test of the export script. For "mid-level" export with `transformers.onnx`, the model was downloaded and quantized but saw the following error:
```
KeyError: "falcon is not supported yet. Only ['albert', 'bart', 'beit', 'bert', 'big-bird', 'bigbird-pegasus', 'blenderbot', 'blenderbot-small', 'bloom', 'camembert', 'clip', 'codegen', 'convbert', 'convnext', 'data2vec-text', 'data2vec-vision', 'deberta', 'deberta-v2', 'deit', 'detr', 'distilbert', 'electra', 'flaubert', 'gpt2', 'gptj', 'gpt-neo', 'groupvit', 'ibert', 'imagegpt', 'layoutlm', 'layoutlmv3', 'levit', 'longt5', 'longformer', 'marian', 'mbart', 'mobilebert', 'mobilenet-v1', 'mobilenet-v2', 'mobilevit', 'mt5', 'm2m-100', 'owlvit', 'perceiver', 'poolformer', 'rembert', 'resnet', 'roberta', 'roformer', 'segformer', 'squeezebert', 'swin', 't5', 'vision-encoder-decoder', 'vit', 'whisper', 'xlm', 'xlm-roberta', 'yolos'] are supported. If you want to support falcon please propose a PR or open up an issue."
```
	 - 10/30/2023 (continued): It seems that the falcon model (along with most of the other llms like flan-t5, llama 2, vicuna, mistral, zephyr, etc) are not yet supported with this method. For "low-level" export with `torch.onnx`, the model was downloaded and quantized but saw the following error:
```
batch_size, num_heads, kv_length, head_dim = past_key_value[0][0].shape
ValueError: not enough values to unpack (expected 4, got 0)
```
	 - 10/30/2023 (continued): This is much more promising than the others. For "high level" export iwth `optimum.onnxruntime`, the model was downloaded and quantized but saw the following error:
```
ValueError: Trying to export a falcon model, that is a custom or unsupported architecture for the task text-generation-with-past, but no custom onnx configuration was passed as `custom_onnx_configs`. Please refer to https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models for an example on how to export custom models. For the task text-generation-with-past, the Optimum ONNX exporter supports natively the architectures: ['bart', 'blenderbot', 'blenderbot_small', 'bloom', 'codegen', 'gpt2', 'gpt_bigcode', 'gptj', 'gpt_neo', 'gpt_neox', 'marian', 'mbart', 'mpt', 'opt', 'llama', 'pegasus'].
```
	 - 10/30/2023 (continued): It is another case of the falcon model not being a part of the list of currently supported models. The same can be replicated if using the `optimum-cli` tool to export the model ([documentation here](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model); the command is `optimum-cli export onnx --model tiiuae/falcon-7b falcon-7b_onnx/`)


### References

 - transformers
	 - [AutoModelForCausalLM](https://huggingface.co/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoModelForCausalLM)
	 - [BitsAndBytesConfig](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)
	 - [Quantization](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/quantization#quantization)
	 - [Quantization concept](https://huggingface.co/docs/text-generation-inference/conceptual/quantization)
	 - [Lora](https://huggingface.co/docs/peft/conceptual_guides/lora)
	 - [PeftConfig](https://huggingface.co/docs/peft/main/en/package_reference/config#peft.PeftConfig)
	 - [Peft GitHub](https://github.com/huggingface/peft)
	 - [Huggingface blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
     - [Huggingface blog on peft](https://huggingface.co/blog/peft)
	 - [Link to QLora paper](https://huggingface.co/papers/2305.14314) on Huggingface
	 - [Optimum inference with ONNX Runtime](https://huggingface.co/docs/optimum/v1.2.1/en/onnxruntime/modeling_ort) on Huggingface
	 - [Huggingface blog on convert transformers to onnx with optimum](https://huggingface.co/blog/convert-transformers-to-onnx)
 - tutorial
	 - [GPT-Neo Video](https://www.youtube.com/watch?v=NRVaRXDoI3g)
	 - [GPT-Neo Collab](https://colab.research.google.com/drive/1Vvju5kOyBsDr7RX_YAvp6ZsSOoSMjhKD?usp=sharing#scrollTo=E0Nl5mWL0k2T)
	 - [Falcon-7B Video](https://www.youtube.com/watch?v=2PlPqSc3jM0)
	 - [Falcon-7B Collab](https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing)
	 - [Medium article](https://medium.com/@srishtinagu19/quantizing-falcon-7b-instruct-for-running-inference-on-colab-bd97066aa49d) to Quantize Falcon 7B Instruct for running inference on colab
	 - [Medium article](https://vilsonrodrigues.medium.com/run-your-private-llm-falcon-7b-instruct-with-less-than-6gb-of-gpu-using-4-bit-quantization-ff1d4ffbabcc) to Run your private LLM: Falcon-7B-Instruct with less than 6GB of GPU using 4-bit quantization
	 - [Medium article](https://medium.com/@amodwrites/a-definitive-guide-to-qlora-fine-tuning-falcon-7b-with-peft-78f500a1f337) on a definitive guide to QLora finetuning of falcon 7b with peft
 - model
	 - [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)
	 - [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b)