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



## Fine-tuning



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

