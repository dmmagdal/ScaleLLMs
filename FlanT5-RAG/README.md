# Flan T5 RAG

Description: This is a quick example of using the Flan T5 model Langchain for retrieval augmented generation. 


### Setup

 - It is recommended that you create a python virtual environment instead of a conda due to version issues with a lot of necessary packages.
 - To set up the virtual environment, install the `venv` package:
	 - `pip3 install virtualenv`
 - Create the new virtual environment:
	 - `python -m venv pt-hf-onnx`
 - Activate the virtual environment:
	 - Linux/MacOS: `source pt-hf-onnx/bin/activate`
	 - Windows: `.\pt-hf-onnx\Scripts\activate`
 - Deactivate the virtual environment:
	 - `deactivate`
 - Install the necessary packages (while the virtual environment is active):
	 - `(pt-hf-onnx) pip3 install -r requirements.txt`
 - Also be sure to install the necessary version of `pytorch` according to your OS (refer to the `pytorch` website but the following command will help):
	 - Linux & Windows (CUDA 11.8): `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
 - Regarding installing `faiss`, there are a few caveats with the library to consider:
     - `faiss` runs only on linux based operating systems (Linux and MacOS). To operate the module on windows, either use Docker or a WSL Linux distro (ie ubuntu)
     - `faiss` has GPU support only for NVIDIA devices. If you are installing the module on a Mac device, install the cpu-only version (`faiss-cpu`) when using either a MacOS or a machine without an NVIDIA GPU


### Notes

 - Exported the model with `optimum-cli`. Command `optimum-cli export onnx --model google/flan-t5-base flan-t5-base-onnx` was used to export the base model of Flan-T5 from Google to ONNX. The same can be done for all the other variants.
     - Currently have all variants of the original flan-t5 model from Google converted to ONNX on huggingface hub. You can find them here (along with their respective size on disk):
         - [dmmagdal/flan-t5-small-onnx](https://huggingface.co/dmmagdal/flan-t5-small-onnx) - 792MB
         - [dmmagdal/flan-t5-base-onnx](https://huggingface.co/dmmagdal/flan-t5-base-onnx) - 2.2GB
         - [dmmagdal/flan-t5-large-onnx](https://huggingface.co/dmmagdal/flan-t5-large-onnx) - 6.5GB
         - [dmmagdal/flan-t5-xl-onnx](https://huggingface.co/dmmagdal/flan-t5-xl-onnx) - 23GB
         - [dmmagdal/flan-t5-xxl-onnx](https://huggingface.co/dmmagdal/flan-t5-xxl-onnx) - Could not export to ONNX (OOM on regular memory during conversion, even on my DarkStar GPU server)
 - Flan-T5 model sizes (according to [this medium article](https://medium.com/@koki_noda/try-language-models-with-python-google-ais-flan-t5-ba72318d3be6)):
     - Flan T5 small - 80M
     - Flan T5 base - 250M
     - Flan T5 large - 780M
     - Flan T5 xl - 3B
     - Flan T5 xxl - 11B


### References

 - transformers
     - [Huggingface T5 Documentation](https://huggingface.co/docs/transformers/model_doc/t5)
     - [Huggingface Flan T5 Documentation](https://huggingface.co/docs/transformers/model_doc/flan-t5)
     - [TransformersJS Guide](https://huggingface.co/docs/transformers.js/custom_usage#convert-your-models-to-onnx): Export a model to ONNX
     - [Transformers Guide](https://huggingface.co/docs/transformers/serialization): Export to ONNX
     - [Optimum Guide](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model): Export a model to ONNX with optimum.exporters.onnx (optimum-cli)
     - [Huggingface Blog](https://huggingface.co/blog/convert-transformers-to-onnx): Convert Transformers to ONNX with Hugging Face Optimum
 - tutorial
     - [Medium Article](https://blog.searce.com/building-a-video-assistant-leveraging-large-language-models-2e964e4eefa1): Building a video assistant leveraging Large Language Models
     - [LinkedIn Post](https://www.linkedin.com/pulse/small-overview-demo-o-google-flan-t5-model-balayogi-g/): A Small Overview and Demo of Google Flan-T5 Model
     - [Medium Article](https://betterprogramming.pub/is-google-flan-t5-better-than-openai-gpt-3-187fdaccf3a6): Is Google’s Flan-T5 Better Than OpenAI GPT-3? (disable javascript on the web page & reload it to bypass the sign in prompt)
 - model
     - [flan-t5-small](https://huggingface.co/google/flan-t5-small)
     - [flan-t5-base](https://huggingface.co/google/flan-t5-base)
     - [flan-t5-large](https://huggingface.co/google/flan-t5-large)
     - [flan-t5-xl](https://huggingface.co/google/flan-t5-xl)
     - [flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)
     - [Xenova/flan-t5-small](https://huggingface.co/Xenova/flan-t5-small) (transformers.js)
     - [Xenova/flan-t5-base](https://huggingface.co/Xenova/flan-t5-base) (transformers.js)
 - references
     - [Medium Article](https://betterprogramming.pub/is-google-flan-t5-better-than-openai-gpt-3-187fdaccf3a6): Is Google’s Flan-T5 Better Than OpenAI GPT-3?
     - [Medium Article](https://medium.com/@koki_noda/try-language-models-with-python-google-ais-flan-t5-ba72318d3be6): Try Language Models with Python: Google AI’s Flan-T5 (premium article)
     - [Narrativa Post](https://www.narrativa.com/flan-t5-a-yummy-model-superior-to-gpt-3/): FLAN-T5, a yummy model superior to GPT-3
     - [Medium Article](https://medium.com/google-cloud/finetuning-flan-t5-base-and-online-deployment-in-vertex-ai-bf099c3a4a86): Fine-tuning Flan-T5 Base and online deployment in Vertex AI
     - [Huggingface Forum](https://discuss.huggingface.co/t/onnx-flan-t5-model-oom-on-gpu/36342): ONNX Flan-T5 Model OOM on GPU
     - [Huggingface Forum](https://discuss.huggingface.co/t/unable-to-import-faiss/3439): Unable to import faiss
     - [Microsoft Open Source Blog](https://cloudblogs.microsoft.com/opensource/2023/10/04/accelerating-over-130000-hugging-face-models-with-onnx-runtime/): Accelerating over 130,000 Hugging Face models with ONNX Runtime