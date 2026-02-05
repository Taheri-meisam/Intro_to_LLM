# LLM Python Project – Setup & Useful Commands

This document lists common commands used to set up, run, and manage a Python-based LLM project on a Linux system (Ubuntu/Debian).

---

## System Update & Dependencies

Update your system and install basic tools:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-venv python3-pip

(Optional but useful)

sudo apt install -y build-essential

Project Setup

Create and enter the project directory:

mkdir LLMproject
cd LLMproject

Clone a repository (if applicable):

git clone <repo_url>
cd <repo_name>

Python Virtual Environment

Create a virtual environment:

python3 -m venv venv

Activate the virtual environment:

source venv/bin/activate

Deactivate the virtual environment:

deactivate

Install Dependencies

Upgrade pip:

pip install --upgrade pip

Install project requirements:

pip install -r requirements.txt

Save installed packages to requirements.txt:

pip freeze > requirements.txt

Running Python Files

Run a Python script:

python main.py

Run a module:

python -m module_name

File & Folder Management

List files and folders:

ls
ls -la

Create a new file:

touch anewfile.py

View file contents:

cat anewfile.py

Open file in editor (nano or vim):

nano anewfile.py
vim anewfile.py

Remove a file:

rm file.py

Remove a folder:

rm -rf afolder

Git Commands

Initialize a repository:

git init

Check status:

git status

Add files:

git add .

Commit changes:

git commit -m "Initial commit"

Pull latest changes:

git pull

Push changes:

git push

LLM / ML Helpful Commands

Check Python version:

python --version

Check installed packages:

pip list

Check GPU availability (NVIDIA):

nvidia-smi

Check CUDA version:

nvcc --version

Test if PyTorch sees GPU:

python -c "import torch; print(torch.cuda.is_available())"

Environment Variables

Set an environment variable (example: API key):

export OPENAI_API_KEY="your_api_key_here"

Verify variable:

echo $OPENAI_API_KEY

Cleanup

Remove virtual environment:

rm -rf venv
