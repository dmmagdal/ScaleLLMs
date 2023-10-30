# bitsandbytes_quantize_falcon.py
# Load the (quantized) Falcon 7B model and finetune it with 
# bitsandbytes.
# Windows/MacOS/Linux


import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer


def main():
	###################################################################
	# Load the model
	###################################################################
	model_id = './bitsandbytes_quantized_4bit_falcon-7b'
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		# quantization_config=bnb_config,			# quantizes the model with the above config
		# device_map={"": 0},						# note that you will need GPU to quantize the model.
		device_map="auto",
		# trust_remote_code=True,
	)

	# We have to apply some preprocessing to the model to prepare it 
	# for training. For that use the prepare_model_for_kbit_training 
	# method from PEFT.
	model.config.use_cache = False
	model = prepare_model_for_kbit_training(model)

	# Define the Lora config for the finetuning.
	config = LoraConfig(
		r=64, 								# the rank of the update matrices. Lower rank results in smaller update matrices with fewer trainable parameters
		lora_alpha=16, 						# LoRA scaling factor
		target_modules=[
			# "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"	# from first example source (causes TypeError: 'NoneType' object is not subscriptable).
			"query_key_value",												# from second example source (no error).
		], # the modules (for example, attention blocks) to apply the LoRA update matrices
		lora_dropout=0.1, 					#
		bias="none", 						# specifies if the bias parameters should be trained (can be "none", "all", or "lora_only")
		task_type="CAUSAL_LM"				# 
	)
	model = get_peft_model(model, config)
	print_trainable_parameters(model)

	tokenizer.pad_token = tokenizer.eos_token

	###################################################################
	# Load the dataset
	###################################################################
	# Using the English quotes dataset.
	data = load_dataset("Abirate/english_quotes")
	data = data.map(
		lambda samples: tokenizer(samples["quote"]), 
		batched=True
	)

	###################################################################
	# Train the model
	###################################################################
	# tokenizer.pad_token = tokenizer.eos_token
	trainer = transformers.Trainer(
		model=model,							# model
		train_dataset=data["train"],			# dataset
		args=transformers.TrainingArguments(	# training args
			per_device_train_batch_size=4,
			gradient_accumulation_steps=4,
			warmup_ratio=0.03,
			max_steps=500,
			max_grad_norm=0.3,
			learning_rate=2e-4,
			logging_steps=10,
			save_steps=10,
			output_dir="results",
			optim="paged_adamw_32bit"
		),
		data_collator=transformers.DataCollatorForLanguageModeling(
			tokenizer, mlm=False
		),
	)

	model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
	trainer.train()

	###################################################################
	# Save and load the finetuned/quantized model
	###################################################################
	model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
	model_to_save.save_pretrained("results-model")

	lora_config = LoraConfig.from_pretrained('results-model')
	model = get_peft_model(model, lora_config)

	###################################################################
	# Sample from the model
	###################################################################
	text = "Elon Musk "
	device = "cuda:0"

	inputs = tokenizer(text, return_tensors="pt").to(device)
	outputs = model.generate(**inputs, max_new_tokens=20)
	print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def print_trainable_parameters(model):
	"""
	Prints the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	for _, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	print(
		f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
	)


if __name__ == '__main__':
	main()