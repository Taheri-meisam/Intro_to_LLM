PYTHON LLM PROJECT – USEFUL COMMANDS

SYSTEM SETUP
------------
Update system and install required tools:

sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip build-essential


PROJECT SETUP
-------------
Create project folder and enter it:
'''bash
mkdir LLMproject
cd LLMproject

Clone a repository (optional):

git clone <repository_url>
cd <repository_name>


PYTHON VIRTUAL ENVIRONMENT
--------------------------
Create virtual environment:

python3 -m venv venv

Activate virtual environment:

source venv/bin/activate

Deactivate virtual environment:

deactivate


DEPENDENCY MANAGEMENT
---------------------
Upgrade pip:

pip install --upgrade pip

Install dependencies:

pip install -r requirements.txt

Save installed packages:

pip freeze > requirements.txt


RUNNING THE PROJECT
-------------------
Run a Python file:

python main.py

Run a module:

python -m module_name


FILE & FOLDER COMMANDS
----------------------
List files:

ls
ls -la

Create a file:

touch anewfile.py

View file contents:

cat anewfile.py

Edit file:

nano anewfile.py
vim anewfile.py

Remove a file:

rm file.py

Remove a folder:

rm -rf afolder


GIT COMMANDS 
------------
Use this only if you can use git or you want to learn how to use git 
Initialize repository:

git init

Check status:

git status

Stage files:

git add .

Commit changes:

git commit -m "Initial commit"

Pull changes:

git pull

Push changes:

git push


LLM / ML UTILITIES
------------------
Check Python version:

python --version

List installed packages:

pip list

Check NVIDIA GPU:

nvidia-smi

Check CUDA version:

nvcc --version

Check PyTorch GPU support:

python -c "import torch; print(torch.cuda.is_available())"



CLEANUP
-------
Remove virtual environment:

rm -rf venv


NOTES
-----
- Always activate the virtual environment before running the project

