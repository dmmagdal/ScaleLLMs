# Scale LLMs

Description: This repo aims to look at different techniques that can allow for LLMs (language models over 1 billion parameters) to be scaled to run and train on consumer grade hardware.


### Notes

 - In this case, consumer grade hardware is referring to desktops that have between 8 to 32 GB of memory (RAM), with an average of 16 GB as well as 8 to 16 GB of GPU memory (VRAM).
	 - My M2 Macbook Pro
		 - OS: MacOS Sonoma
		 - Apple M2 Silicon
		 - 16 GB RAM
		 - 8 GB VRAM
	 - My Dell XPS Desktop
		 - OS: Windows 11
		 - Intel i7 CPU
		 - 16 GB RAM
		 - Nvidia 2060 SUPER (8 GB VRAM)
	 - My Darkstar GPU server
		 - OS: Ubuntu 22.04
		 - Dual Intel Xeon CPUs
		 - 64 GB RAM
		 - 3x Nvidia Tesla P100 (16 GB VRAM each)
 - Personal environment notes (software setup):
	 - Conda's conda-forge library does not currently have the correct version of `auto-gptq` or `peft` available
	 - In addition, the conda-forge version of `bitsandbytes` may not be correct or up to date (was having an error where upon loading the module, `bitsandbytes` could not find CUDA)
	 - To counter this, use python's `virtualenv` package ([documentation here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/))
		 - install: `pip install virtualenv`
		 - create new virtual environment: `python3 -m venv env-name`
		 - activate virtual environment: `source env-name/bin/activate`
		 - deactivate virtual environment: `deactivate`
		 - install packages to virtual environment (while environment is active): `pip install package-name`
		 - would be best to deactivate any/all existing conda enviornments before doing anything in python's virtualenv
	 - Install `pytorch` (see [website](https://pytorch.org/) for most up to date commands):
		 - Linux (CUDA 11.8):
			 - `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
			 - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
		 - MacOS (MPS acceleration on MacOS 12.3+): 
			 - `conda install pytorch::pytorch torchvision torchaudio -c pytorch`
			 - `pip install torch torchvision torchaudio`
		 - Windows (CUDA 11.8):
			 - `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
			 - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
	 - To alleviate confusion between conda and `virtualenv`:
		 - Use `requirements.txt` to initialize the virtual environment.
		 - Use `environment.yaml` to initialize the conda environment.
 - Techniques to be explored in this repo:
	 - QLora (with `bitsandbytes` library in addition to huggingface's `transformers` & `peft`)
	 - GPTQ (with `auto-gptq` and/or `optimum` libraries in addition to huggingface's `transformers` & `peft`)
	 - GGML/GGUF
 - QLora
	 - `bitsandbytes` requires CUDA GPU to do quantization.
	 - `bitsandbytes` benefits:
		 - Can peform "zero-shot quantization" (does not require input data to calibrate the quantized model)
		 - Can quantize any model out of the box as long as it contains `torch.nn.Linear` modules 
		 - Quantization is peformed on model load, no need to run any post-processing or preparation step
		 - Zero peformance degredation when loading adapters and can also merge adapters in a dequantized model
	 - `bitsandbytes` shortcomings:
		 - 4-bit quantization not serializable (means unable to save 4-bit quantized models).
 - GPTQs
	 - `auto-gptq` requires CUDA or RoCM (AMD) GPU to do quantization.
	 - `auto-gptq` benefits:
		 - fast for text generation
		 - n-bit support
		 - easily serializable
	 - `auto-gptq` shortcomings:
		 - requires a calibration dataset
		 - works for language models only (at the time of writing this 10/10/2023)
 - GGML/GGUF
	 - Models quantized with GGML/GGUF [do NOT currently support any form of finetuning](https://github.com/ggerganov/ggml/issues/8) (and are therefore limited to just inference).
 - AWQ
	 - `autoawq` requires CUDA GPU to do quantization
 - Can finetune quantized models with `peft` library from huggingface (for GPTQ and QLora quantization)
	 - PEFT stands for "parameter efficient finetuning".
	 - Training on quantized models is not possible.
 - Overall notes on my experience with quantization and finetuning
	 - The reliance on (primarily CUDA) GPUs is a detriment to the democratization of AI models. Current quantization methods (QLora and GPTQ) do a lot to allow for LLMs to be fine tuned and run on consumer desktops. However, not everyone has the budget for a GPU with the necessary VRAM to perform the quantization (looks to be 12-16+ GB VRAM for things like a 7b parameter model) on LLMs. This is only exasterbated by the reliance on NVIDIA cards. I understand `auto-gptq` allows for RoCm AMD cards to work as well but there is a blatant emphasis on CUDA support. Apple silicon is not viable for any quantization package at this time (10/21/2023) as support for MPS is not included in any of the major packages (`bitsandbytes` and `auto-gptq`), which cuts out another group of potential users (especially as Apple puts more effort into creating powerful hardware like their mac studio or mac station).
	 - Speaking of neglected users, Windows users have to rely on other packages and work arounds to get `bitsandbytes` working for their environments. While the community has banded together to come up with the necessary 
	 - Quantizing Falcon 7B (with `auto-gptq`) took more resources than my Dell XPS Desktop was able to give (would usually OOM for the GPU). On Colab instances (free tier for Google and Kaggle) there would be other issues concerning not enough memory (regular RAM) or setting up the environment with the necessary modules (Kaggle was having a weird time importing/using `auto-gptq` despite the import command being run several times).
		 - Note on Colab and Kaggle free tier instances:
			 - Colab free tier has 12.7GB RAM and 16GB VRAM (Nvidia T4).
			 - Kaggle free tier has 29GB RAM and 2x 16GB VRAM (Nvidia T4).
	 - Why learn how to quantize a model when others are already doing it?
		 - It's good to know how things work.
		 - You have more control over your models (other people's models/copies can be faulty, corrupted, tampered, or censored).
	 - What should you do if you cannot quantize a model (ie package error messages, insufficient hardware, etc)?
		 - Wait. Waiting for package updates and tutorials gives you the benefit of having someone else go through the trouble of debugging or handling your specific use case.
		 - File a bug. In the case of errors from running your quantization + finetuning, there are probably a lot of other people who have encountered the same bug on their system. Keeping up to date or subscribed to those posts may give you either a work around or a note that your issue is going to be resolved in the next update.
		 - Use a model that's already been quantized. While I mentioned above why *you* may want to perform your own quantization, there are still many people in the community who have gone through the trouble of quantizing the desired model for you (such as [TheBloke](https://huggingface.co/TheBloke)). You will still have to judge how reputable/trustworthy those files are (even if they're using `safetensors`) but there are generally a lot of good actors in the community hoping to enable others to continue their research or work with different large scale models (like LLMs or Stable Diffusion).
		 - Find another way/model. There are now many versions of models that come out, each version the same architecture but with a different number of parameters (usually as a result of using different number of layers and hyperparameters). Using smaller or distilled models at the cost of accuracy is still a viable option for your work. While this isnt quantization, it can be of tremendous benefit to you. You can also look to see if there is another way of doing things such as quantization. This repo explores QLora, GPTQ, and GGML/GGUF, but is primarily focused on the first two since the last one doesnt allow for finetuning after model quantization. While newer and more efficient methods will be developed, these three give you many options for how to quantize large models.


### References

 - YouTube video from Code_Your_Own_AI: [New Tutorial on LLM Quantization w/ QLoRA, GPTQ and Llamacpp, LLama 2](https://www.youtube.com/watch?v=YEVyupJxt1Q&ab_channel=code_your_own_AI)
 - Huggingface Transformers [documentation on quantization](https://huggingface.co/docs/text-generation-inference/conceptual/quantization)
 - Huggingface Transformers [overview of quantization](https://github.com/huggingface/blog/blob/main/overview-quantization-transformers.md) on GitHub
 - Huggingface Optimum [documentation on quantization](https://huggingface.co/docs/optimum/concept_guides/quantization)
 - Huggingface Transformers [documentation on quantization](https://huggingface.co/docs/transformers/main_classes/quantization)
 - Huggingface Accelerate [documentation on estimating model footprint on memory](https://huggingface.co/docs/accelerate/v0.23.0/en/usage_guides/model_size_estimator)