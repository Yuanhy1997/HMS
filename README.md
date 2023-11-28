# Training and Inference of Large Language Models (LLMs) for Clinical Purposes

## The Model List:

For LLMs:
   1. [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
   2. [WizardLM/WizardLM-7B-V1.0](https://huggingface.co/WizardLM/WizardLM-7B-V1.0)
   3. [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
   4. [lmsys/vicuna-7b-v1.5-16k](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k)

For SentenceBERT:
   1. [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
   2. [sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)
   3. [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) ## IF using bge, activate normalized embeddings in sentencebert: model.encode(sentences_1, normalize_embeddings=True)
   4. [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

For ColBERT:
   The checkpoint can be downloaded from [here](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz).

## Inference

To start with, current open-source LLMs mainly based on PyTorch and HuggingFace Hubs. The first step is to set up environments:

1. Download Anaconda for better management of virtual environments. To download and install, using the following line in the terminal:
   ```{bash}
   cd <where-to-download>
   wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   bash Anaconda3-2023.09-0-Linux-x86_64.sh
   ```
2. After installation, we can create an virtual environment by:
   ```{bash}
   conda create -n inference python=3.10
   ```
   then we can go into this environment by:
   ```{bash}
   conda activate inference
   ```
3. We can set the python environment with pip:
   ```{bash}
   pip install -r requirements.txt
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
   pip install "fschat[model_worker,webui]"
   pip install -U sentence-transformers
   pip install -U FlagEmbedding
   ```

<!-- For runing inference, we have to prepare two things locally:
1. Local LLM Weights, if the nodes can have access to the internet, we can also use the online HuggingFace Model Hubs.
2. Local File for inference, to fit this code, the file have to be reformatted to a .jsonl file, in which each line presents one sample and is formatted as:
   ```
   {'sample_idx': <sample_idx>, 'instruction': <input_query>}
   ```
   Basically, we can use json package in Python to generate such formats. We have an code example [here](./inference/generate_dummydata.py).


Here can we start inference! Using these lines:
```{bash}
cd <path-to-this-repo>/inference/
bash ./llm_inference.sh <local_model_path> <input_jsonl_file> <output_file_directory_name> <number-of-gpu-to-use>
```
Then we can check the output jsonl file in output/<output_file_directory_name>/results.jsonl.
 -->


<!-- ## Fine-tuning



To start with, current open-source LLMs mainly based on PyTorch and HuggingFace Hubs. The first step is to set up environments:

1. Download Anaconda for better management of virtual environments. To download and install, using the following line in the terminal:
   ```{bash}
   cd <where-to-download>
   wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   bash Anaconda3-2023.09-0-Linux-x86_64.sh
   ```
2. After installation, we can create an virtual environment by:
   ```{bash}
   conda create -n llm_inference python=3.10
   ```
   then we can go into this environment by:
   ```{bash}
   conda activate llm_finetune
   ```
3. We can set the python environment with pip:

   *Note: Below is the instruction for A100 for Vidul*
   ```{bash}
   conda install cudatoolkit-dev -c conda-forge
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
   cd <path-to-this-repo>/fine-tune/
   pip install -r requirements.txt
   ```
   Then we have to install flash-attention for accelerated training, [flash-attn](https://github.com/Dao-AILab/flash-attention): 
   To install:
   1. Make sure that PyTorch is installed.
   2. Make sure that `packaging` is installed (`pip install packaging`)
   3. Make sure that `ninja` is installed and that it works correctly (e.g. `ninja
   --version` then `echo $?` should return exit code 0). If not (sometimes `ninja
   --version` then `echo $?` returns a nonzero exit code), uninstall then reinstall
   `ninja` (`pip uninstall -y ninja && pip install ninja`). Without `ninja`,
   compiling can take a very long time (2h) since it does not use multiple CPU
   cores. With `ninja` compiling takes 3-5 minutes on a 64-core machine.
   4. Then:
   ```sh
   pip install flash-attn --no-build-isolation
   ```
   Alternatively you can compile from source:
   ```sh
   python setup.py install
   ```


   *Note: Below is the instruction for V100 for Sara*
   ```{bash}
   conda install cudatoolkit-dev -c conda-forge
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
   cd <path-to-this-repo>/fine-tune/
   pip install -r requirements.txt
   ```
   



After setting up environments, for runing fine-tuning, we also have to prepare two things locally:
1. Local LLM Weights, if the nodes can have access to the internet, we can also use the online HuggingFace Model Hubs.
2. Local File for fine-tuning, to fit this code, the file have to be reformatted to a .jsonl file, in which each line presents one sample and is formatted as:
   ```
   {'sample_idx': <sample_idx>, 'instruction': <input_query>, 'output': <input_query>}
   ```
   Basically, we can use json package in Python to generate such formats. We have an code example [here](./inference/generate_dummydata.py).

*It should be noted that: Flash Attention which is a framework for fast fine-tuning with higher speed and low memory cost does not support V100. Therefore for A100, you can use llama_train.sh or llama2_train.sh for fine-tuning, which depends on the LLM you base on. For V100, you can use main_llama_noflash.sh for both llama and llama2. And This is why we have different env set up for Vidul and Sara.* 

Here can we start fine-tuning! Using these lines:
```{bash}
cd <path-to-this-repo>/fine-tune/
bash ./llama_train.sh <local_model_path> <input_jsonl_file> <output_file_directory_name> <number-of-gpu-to-use>
```
Then we can check the fine-tuned model in output/<output_file_directory_name>/




## SentenceBERT

Setting up environments:

1. Download Anaconda for better management of virtual environments. To download and install, using the following line in the terminal:
   ```{bash}
   cd <where-to-download>
   wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   bash Anaconda3-2023.09-0-Linux-x86_64.sh
   ```
2. After installation, we can create an virtual environment by:
   ```{bash}
   conda create -n sentencebert python=3.10
   ```
   then we can go into this environment by:
   ```{bash}
   conda activate sentencebert
   ```
3. We can set the python environment with pip:
   ```{bash}
   pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
   cd <path-to-this-repo>/sentenceBERT/
   pip install -r requirements.txt
   pip install -U sentence-transformers
   ```

After Setting Up environment, we have to download the embedding model locally or use the Huggingface Hub. The accessible models are listed [here](https://www.sbert.net/docs/pretrained_models.html#). 

Then prepare all your sentences or terms for embeddings to a jsonl file with each line, here is an example:
```
{'input': 'harvard medical school'}
```

Finally, we can generate the sentence embedding by:
```{bash}
cd <path-to-this-repo>/sentenceBERT/
CUDA_VISIBLE_DEVICES=0 python main.py <jsonl-file-of-sentences> <local-model-path> <output-dir-name>
```
The sentence embeddings will be saved to a .pkl file in './sentenceBERT/output/output-dir-name/embeddings.pkl'

To load the saved sentence embedding for other use, you can run the following lines in a Python:
```{python}
import pickle
with open(<path-to-embeddings.pkl>, "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']
```
 -->

