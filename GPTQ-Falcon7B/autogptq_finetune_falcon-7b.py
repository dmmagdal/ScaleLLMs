# autogptq_finetune_falcon-7b.py
# Load the quantized Falcon 7B model and finetune it with Peft.
# Windows/MacOS/Linux


import os
import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTQConfig
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training


def main():
	###################################################################
	# Load the quantized model
	###################################################################

	# You can load models that have been quantized using the auto-gptq 
	# library out of the box from the 🤗 Hub directly using 
	# from_pretrained method. Make sure that the model on the Hub have 
	# an attribute quantization_config on the model config file, with 
	# the field quant_method set to "gptq".

	# Most used quantized models can be found under TheBloke namespace 
	# that contains most used models converted with auto-gptq library. 
	# The integration should work with most of these models out of the 
	# box (to confirm and test).

	# Check for local copy of model, otherwise load from huggingface 
	# hub.
	model_id = "falcon-7b-gptq-4bit"
	local_path = "./autogptq_quantized_4bit_falcon-7b"
	if os.path.exists(local_path):
		tokenizer = AutoTokenizer.from_pretrained(
			local_path,
			local_files_only=True
		)
		model = AutoModelForCausalLM.from_pretrained(
			local_path,
			local_files_only=True,
			device_path="auto",
		)
	else:
		quantization_config = GPTQConfig(
			bits=4,
			disable_exllama=True,
		)
		tokenizer = AutoTokenizer.from_pretrained(
			model_id,
		)
		model = AutoModelForCausalLM.from_pretrained(
			model_id,
			quantization_config=quantization_config,
			device_map="auto",
		)

	# Below we will load a llama 7b quantized in 4bit.
	# model_id = "falcon-7b-gptq-4bit"
	# model = AutoModelForCausalLM.from_pretrained(
	# 	model_id, 
	# 	device_map="auto",
	# )
	# tokenizer = AutoTokenizer.from_pretrained(model_id)

	# Once tokenizer and model has been loaded, let's generate some 
	# text. Before that, we can inspect the model to make sure it has 
	# loaded a quantized model. As you can see, linear layers have been
	# modified to QuantLinear modules from auto-gptq library.
	print(model)

	# Furthermore, we can see that from the quantization config that we
	# are using exllama kernel (disable_exllama = False). Note that it 
	# only works with 4-bit model.
	print(model.config.quantization_config.to_dict())

	text = "Hello my name is"
	inputs = tokenizer(text, return_tensors="pt").to(0)

	out = model.generate(**inputs, max_new_tokens=50)
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
	quantization_config_loading = GPTQConfig(
		bits=4, 
		disable_exllama=True,
	)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		quantization_config=quantization_config_loading, 
		# device_map="auto",
		device_map={"": 0},		# use on my Dell Desktop
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

	peft_model = get_peft_model(model, config)
	peft_model.print_trainable_parameters()

	# Finally, let's load a dataset and we can train our model.
	data = load_dataset("Abirate/english_quotes")
	data = data.map(
		lambda samples: tokenizer(samples["quote"]), 
		batched=True
	)

	# needed for llama 2 tokenizer
	# tokenizer.pad_token = tokenizer.eos_token

	# Initialize trainer and train the model.
	trainer = Trainer(
		model=peft_model,
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
	peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
	trainer.train()

	# AutoGPTQ library offers multiple advanced options for users that 
	# wants to explore features such as fused attention or triton 
	# backend. Therefore, we kindly advise users to explore AutoGPTQ 
	# library for more details.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()