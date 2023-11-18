# bitsandbytes_quantize_falcon.py
# Load the (quantized) Falcon 7B model and finetune it with 
# bitsandbytes.
# Windows/MacOS/Linux


from argparse import ArgumentParser
import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer


def main():
	# Parse arguments.
	parser = ArgumentParser(
		description="Finetune the quantized falcon 7b model"
	)
	parser.add_argument(
		'--quantize_4_bit',
		action='store_true',
		default=False,
		help="load 4-bit quantized model"
	)
	args = parser.parse_args()

	if args.quantize_4_bit:
		print("Saving 4-bit quantized model is not currently supported in transformers.")
		print("No model/tokenizer was saved.")
		exit(0)

	###################################################################
	# Load the model
	###################################################################
	bits = "4bit" if args.quantize_4_bit else "8bit"
	model_id = f'./bitsandbytes_quantized_{bits}_falcon-7b'
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		# device_map={"": 0},						# note that you will need GPU to quantize the model.
		device_map="auto",
	)
	print(model)

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
			# max_steps=500,					# old max_steps
			max_steps=100,						# new max_steps (wanted something sooner)
			max_grad_norm=0.3,
			learning_rate=2e-4,
			logging_steps=10,
			save_steps=10,
			output_dir=f"{bits}-results",
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
	save = f"{bits}-falcon7b-results-model"
	model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
	model_to_save.save_pretrained(save)

	lora_config = LoraConfig.from_pretrained(save)
	model = get_peft_model(model, lora_config)

	###################################################################
	# Sample from the model
	###################################################################
	text = "Elon Musk "
	device = "cuda:0"

	inputs = tokenizer(text, return_tensors="pt").to(device)
	outputs = model.generate(**inputs, max_new_tokens=20)
	print(tokenizer.decode(outputs[0], skip_special_tokens=True))

	# Exit the program.
	exit(0)


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