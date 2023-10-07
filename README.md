# Training and Inference of Large Language Models (LLMs) for Clinical Purposes

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
   conda create -n llm_inference python=3.10
   ```
   then we can go into this environment by:
   ```{bash}
   conda activate llm_inference
   ```
3. We can set the python environment with pip:
   ```{bash}
   ## For Vidul
   pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
   ## For Sara
   pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102  ## For Sara
   cd <path-to-this-repo>/inference/
   pip install "fschat[model_worker,webui]"
   pip install -r requirements.txt
   ```

For runing inference, we have to prepare two things locally:
1. Local LLM Weights, if the nodes can have access to the internet, we can also use the online HuggingFace Model Hubs.
2. Local File for inference, to fit this code, the file have to be reformatted to a .jsonl file, in which each line presents one sample and is formatted as:
   ```
   {'sample_idx': <sample_idx>, 'query': <input_query>}
   ```
   Basically, we can use json package in Python to generate such formats. We have an code example (here)[].


Here can we start inference! Using these lines:
```{bash}
cd <path-to-this-repo>/inference/
bash ./llm_inference.sh <local_model_path> <input_jsonl_file> <output_file_directory_name>
```
Then we can check the output jsonl file in output/<output_file_directory_name>/results.jsonl.


