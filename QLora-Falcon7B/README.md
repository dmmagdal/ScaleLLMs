# QLora Falcon 7B

Description: This is a quick example of finetuning the Falcon 7B model with QLora given a set of text.


### Notes

 - LLM models I want to quantize:
	 - Falcon 1.3B and/or 7B (maybe do 40B if I have the resources)
	 - Llama 2 7B and/or 13B (most likely dont have enough resources for 70B)
	 - T5 (an older model whose uses I have yet to understand but has great zero shot versatility from what I've heard)
	 - Flan-T5 (let's see how the ChatGPT killer really performs)
 - As of writing this (10/07/2023), I have been able to download the `bitsandbytes` module required for quantizing models. However, upon trying to run the programs, I get an error from `bitsandbytes` regarding the module not being able to locate the CUDA version. Going to GitHub, I've seen this is an issue with other users and the repo owner has marked such issues as `[CUDA_SETUP]` and `[low-priority]`. I have low expectations as to whether this will be updated anytime soon.
	 - I am considering alternatives such as GGML/GGUL and GPTQ for quantizing the models. Actual finetuning may be different as a result but I have to get to that part next.


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
 - tutorial
	 - [GPT-Neo Video](https://www.youtube.com/watch?v=NRVaRXDoI3g)
	 - [GPT-Neo Collab](https://colab.research.google.com/drive/1Vvju5kOyBsDr7RX_YAvp6ZsSOoSMjhKD?usp=sharing#scrollTo=E0Nl5mWL0k2T)
	 - [Falcon-7B Video](https://www.youtube.com/watch?v=2PlPqSc3jM0)
	 - [Falcon-7B Collab](https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing)
	 - [Medium article](https://medium.com/@srishtinagu19/quantizing-falcon-7b-instruct-for-running-inference-on-colab-bd97066aa49d) to Quantize Falcon 7B Instruct for running inference on colab
	 - [Medium article](https://vilsonrodrigues.medium.com/run-your-private-llm-falcon-7b-instruct-with-less-than-6gb-of-gpu-using-4-bit-quantization-ff1d4ffbabcc) to Run your private LLM: Falcon-7B-Instruct with less than 6GB of GPU using 4-bit quantization
 - model
	 - [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)
	 - [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b)