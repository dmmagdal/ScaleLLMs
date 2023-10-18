# gptq_llama2.py
# Load the Llama 2 7B model and finetune it with GPTQ from 
# huggingface's Optimum library.
# Windows/MacOS/Linux


import torch
from accelerate import init_empty_weights
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTQConfig
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from optimum.gptq import GPTQQuantizer, load_quantized_model
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training


def main():
	###################################################################
	# Load the model
	###################################################################

	# Define model for download and the (auto) gptq config for the
	# quantization step.
	model_id = "facebook/opt-125m"

	# Initialize model tokenizer and the model.
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		device_map="auto",					# note that you will need GPU to quantize the model.
		# device_map={"": 0},
		trust_remote_code=True,
		torch_dtype=torch.float16,
	)
	
	quantizer = GPTQQuantizer(
		bits=4,											# the number of bits to quantize to, supported numbers are (2, 3, 4, 8).
		dataset="c4",									# the dataset used for quantization. You can provide your own dataset in a list of string or just use the original datasets used in GPTQ paper [‘wikitext2’,‘c4’,‘c4-new’,‘ptb’,‘ptb-new’]
		block_name_to_quantize="model.decoder.layers",	# the block name to quantize
		model_seqlen=2048,								# the model sequence length used to process the dataset
	)

	# Quantize the model.
	quant_model = quantizer.quantize_model(model, tokenizer)

	# You can make sure the model has been correctly quantized by 
	# checking the attributes of the linear layers, they should contain
	# qweight and qzeros attributes that should be in torch.int32 
	# dtype.
	print(model.model.decoder.layers[0].self_attn.q_proj.__dict__)

	# Now let's perform an inference on the quantized model. Use the 
	# same API as transformers!
	text = "Hello my name is"
	inputs = tokenizer(text, return_tensors="pt").to(0)

	# out = model.generate(**inputs)
	out = quant_model.generate(**inputs)
	print(tokenizer.decode(out[0], skip_special_tokens=True))

	###################################################################
	# Save the model
	###################################################################

	# se the save method from GPTQQuantizer class. It will create a 
	# folder with your model state dict along with the quantization 
	# config.
	save_folder = "./gptq_quantized_opt-125m"
	quantizer.save(model, save_folder)

	###################################################################
	# Load the quantized model
	###################################################################

	# You can load your quantized weights by using the 
	# load_quantized_model() function. Through the Accelerate library, 
	# it is possible to load a model faster with a lower memory usage. 
	# The model needs to be initialized using empty weights, with 
	# weights loaded as a next step.
	with init_empty_weights():
		empty_model = AutoModelForCausalLM.from_pretrained(
			model_id, 
			torch_dtype=torch.float16
		)
	empty_model.tie_weights()

	# Exllama kernels for faster inference
	# For 4-bit model, you can use the exllama kernels in order to a 
	# faster inference speed. It is activated by default. If you want 
	# to change its value, you just need to pass disable_exllama in 
	# load_quantized_model(). In order to use these kernels, you need 
	# to have the entire model on gpus.
	quantized_model = load_quantized_model(
		empty_model, 
		save_folder=save_folder,
		device_map="auto",
		disable_exllama=False,	# Note that only 4-bit models are supported with exllama kernels for now. Furthermore, it is recommended to disable the exllama kernel when you are finetuning your model with peft.
	)

	# Once tokenizer and model has been loaded, let's generate some 
	# text. Before that, we can inspect the model to make sure it has 
	# loaded a quantized model. As you can see, linear layers have been
	# modified to QuantLinear modules from auto-gptq library.
	# print(model)
	print(quantized_model)

	# Furthermore, we can see that from the quantization config that we
	# are using exllama kernel (disable_exllama = False). Note that it 
	# only works with 4-bit model.
	# print(model.config.quantization_config.to_dict())
	# print(quantized_model.config.quantization_config.to_dict())

	text = "Hello my name is"
	inputs = tokenizer(text, return_tensors="pt").to(0)

	# out = model.generate(**inputs, max_new_tokens=50)
	out = quantized_model.generate(**inputs, max_new_tokens=50)
	print(tokenizer.decode(out[0], skip_special_tokens=True))

	###################################################################
	# Train/Finetune the quantized model
	###################################################################

	# Let's train the llama-2 model using PEFT library from Hugging 
	# Face 🤗. We disable the exllama kernel because training with 
	# exllama kernel is unstable. To do that, we pass a GPTQConfig 
	# object with disable_exllama=True. This will overwrite the value 
	# stored in the config of the model.
	model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
	quantizer = GPTQQuantizer(
		bits=4, 
		dataset="c4",
		# block_name_to_quantize="model.decoder.layers",
		model_seqlen=2048,
		disable_exllama=True,
	)
	model = AutoModelForCausalLM.from_pretrained(
		model_id, 
		device_map="auto",
		torch_dtype=torch.float16,
	)

	model = quantizer.quantize_model(
		model, tokenizer
	)

	print(model.config.quantization_config.to_dict())

	# First, we have to apply some preprocessing to the model to 
	# prepare it for training. For that, use the 
	# prepare_model_for_kbit_training method from PEFT.
	model.gradient_checkpointing_enable()
	model = prepare_model_for_kbit_training(model)

	# Then, we need to convert the model into a peft model using 
	# get_peft_model.
	config = LoraConfig(
		r=8,
		lora_alpha=32,
		target_modules=["k_proj","o_proj","q_proj","v_proj"],
		lora_dropout=0.05,
		bias="none",
		task_type="CAUSAL_LM"
	)

	model = get_peft_model(model, config)
	model.print_trainable_parameters()

	# Finally, let's load a dataset and we can train our model.
	data = load_dataset("Abirate/english_quotes")
	data = data.map(
		lambda samples: tokenizer(samples["quote"]), 
		batched=True
	)

	# needed for llama 2 tokenizer
	tokenizer.pad_token = tokenizer.eos_token

	trainer = Trainer(
		model=model,
		train_dataset=data["train"],
		args=TrainingArguments(
			per_device_train_batch_size=1,
			gradient_accumulation_steps=4,
			warmup_steps=2,
			max_steps=10,
			learning_rate=2e-4,
			fp16=True,
			logging_steps=1,
			output_dir="outputs",
			optim="adamw_hf",
		),
		data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
	)
	model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
	trainer.train()

	# AutoGPTQ library offers multiple advanced options for users that 
	# wants to explore features such as fused attention or triton 
	# backend. Therefore, we kindly advise users to explore AutoGPTQ 
	# library for more details.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()