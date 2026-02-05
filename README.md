#  Python LLM Project – Useful Commands

A practical cheat sheet for setting up, managing, and running a **Python-based LLM project** on Linux (Ubuntu/Debian).

---

##  System Setup

Update the system and install required tools:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip build-essential
```
## Project Setup

Create a project folder and enter it:

mkdir LLMproject
cd LLMproject

Clone a repository (optional):

git clone <repository_url>
cd <repository_name>

### Python Virtual Environment

Create a virtual environment:

python3 -m venv venv

Activate the virtual environment:

source venv/bin/activate

Deactivate the virtual environment:

deactivate

Dependency Management

Upgrade pip:

pip install --upgrade pip

Install dependencies:

pip install -r requirements.txt

Save installed packages:

pip freeze > requirements.txt

Running the Project

Run a Python file:

python main.py

File & Folder Commands

List files:

ls
ls -la

Create a file:

touch anewfile.py

View file contents:

cat anewfile.py

Edit a file:

nano anewfile.py
# or
vim anewfile.py

Remove a file:

rm file.py

Remove a folder:

rm -rf afolder

Git Commands

Use this section if you want to use or learn Git:

git init
git status
git add .
git commit -m "Initial commit"
git pull
git push

LLM / ML Utilities

Useful checks for Python, packages, and GPU support:

python --version
pip list
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

🧹 Cleanup

Remove the virtual environment:

rm -rf venv
