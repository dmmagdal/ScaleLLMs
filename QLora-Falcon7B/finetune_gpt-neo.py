# finetune_gpt-neo.py
# Load the GPT NeoX 20B model and finetune it with QLora.
# Windows/MacOS/Linux


import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model


def main():
	###################################################################
	# Load the model
	###################################################################

	# Define model for download and the bitsandbytes config for the
	# quantization step.
	model_id = "EleutherAI/gpt-neox-20b"		# Note that the target_module specified below in LoraConfig will only work with this model
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,						# enable 4bit quantization by replacing the linear layers with fp4/nf4 layers from bitsandbytes
		bnb_4bit_use_double_quant=True,			# flag used for nested quantization where quantization constants from the first quantization are quantized again
		bnb_4bit_quant_type='nf4',				# sets the quantization data type in the bnb.nn.Linear4Bit layers (either fp4 or np4)
		bnb_4bit_compute_dtype=torch.bfloat16,	# sets the computational type which might be different than the input type
	) # Note that np4 stands for normal point 4 bit 

	# Initialize model tokenizer and the model.
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		quantization_config=bnb_config,			# quantizes the model with the above config
		device_map={"": 0},						# note that you will need GPU to quantize the model.
	)

	# We have to apply some preprocessing to the model to prepare it 
	# for training. For that use the prepare_model_for_kbit_training 
	# method from PEFT.
	model.gradient_checkpointing_enable()
	model = prepare_model_for_kbit_training(model)

	# Define the Lora config for the finetuning.
	config = LoraConfig(
		r=8, 								# the rank of the update matrices. Lower rank results in smaller update matrices with fewer trainable parameters
		lora_alpha=32, 						# LoRA scaling factor
		target_modules=["query_key_value"], # the modules (for example, attention blocks) to apply the LoRA update matrices
		lora_dropout=0.05, 					#
		bias="none", 						# specifies if the bias parameters should be trained (can be "none", "all", or "lora_only")
		task_type="CAUSAL_LM"				# 
	)
	model = get_peft_model(model, config)
	print_trainable_parameters(model)

	###################################################################
	# Load the dataset
	###################################################################

	# Using the English quotes dataset.
	data = load_dataset("Abirate/english_quotes")
	data = data.map(
		lambda samples: tokenizer(samples["quote"]), batched=True
	)

	###################################################################
	# Train the model
	###################################################################

	tokenizer.pad_token = tokenizer.eos_token
	trainer = transformers.trainer(
		model=model,							# model
		train_dataset=data["train"],			# dataset
		args=transformers.TrainingArguments(	# training args
			per_device_train_batch_size=1,
			gradient_accumulation_steps=4,
			warmup_steps=2,
			max_steps=10,
			learning_rate=2e-4,
			fp16=True,
			logging_steps=1,
			output_dir="outputs",
			optim="paged_adamw_8bit"
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
	model_to_save.save_pretrained("output-model")

	lora_config = LoraConfig.from_pretrained('output-model')
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