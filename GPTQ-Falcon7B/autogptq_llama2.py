# autogptq_llama2.py
# Load the Llama 2 7B model and finetune it with AutoGPTQ.
# Windows/MacOS/Linux


import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTQConfig
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
# from peft import prepare_model_for_int8_training
from peft import prepare_model_for_kbit_training


def main():
	###################################################################
	# Load the model
	###################################################################

	# Define model for download and the (auto) gptq config for the
	# quantization step.
	model_id = "facebook/opt-125m"
	gptq_config =GPTQConfig(
		bits=4,						# enable 4bit quantization by replacing the linear layers with fp4/nf4 layers from bitsandbytes
		group_size=128,
		dataset="c4",
		desc_act=False,
	)

	# Initialize model tokenizer and the model.
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		quantization_config=gptq_config,		# quantizes the model with the above config
		device_map="auto",						# note that you will need GPU to quantize the model.
		trust_remote_code=True,
	)

	# You can make sure the model has been correctly quantized by 
	# checking the attributes of the linear layers, they should contain
	# qweight and qzeros attributes that should be in torch.int32 
	# dtype.
	print(model.model.decoder.layers[0].self_attn.q_proj.__dict__)

	# Now let's perform an inference on the quantized model. Use the 
	# same API as transformers!
	tokenizer = AutoTokenizer.from_pretrained(model_id)

	text = "Hello my name is"
	inputs = tokenizer(text, return_tensors="pt").to(0)

	out = model.generate(**inputs)
	print(tokenizer.decode(out[0], skip_special_tokens=True))

	###################################################################
	# Quantize with GPTQ on a custom dataset
	###################################################################

	# You can also quantize a model by passing a custom dataset, for 
	# that you can provide a list of strings to the quantization 
	# config. A good number of sample to pass is 128. If you do not 
	# pass enough data, the performance of the model will suffer.
	quantization_config = GPTQConfig(
		bits=4,
		group_size=128,
		desc_act=False,
		dataset=["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."],
	)

	tokenizer = AutoTokenizer.from_pretrained(model_id)
	quant_model = AutoModelForCausalLM.from_pretrained(
		model_id, 
		quantization_config=quantization_config, 
		torch_dtype=torch.float16, 
		device_map="auto"
	)

	# As you can see from the generation below, the performance seems 
	# to be slightly worse than the model quantized using the c4 
	# dataset.
	text = "My name is"
	inputs = tokenizer(text, return_tensors="pt").to(0)

	out = quant_model.generate(**inputs)
	print(tokenizer.decode(out[0], skip_special_tokens=True))

	###################################################################
	# Upload the model
	###################################################################

	# After quantizing the model, it can be used out-of-the-box for 
	# inference or you can push the quantized weights on the 🤗 Hub to 
	# share your quantized model with the community.
	# quant_model.push_to_hub("opt-125m-gptq-4bit")
	# tokenizer.push_to_hub("opt-125m-gptq-4bit")

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

	# Below we will load a llama 7b quantized in 4bit.
	model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
	model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
	tokenizer = AutoTokenizer.from_pretrained(model_id)

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
		device_map="auto",
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
	data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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