╔════════════════════════════════════════════════════╗
║        PYTHON LLM PROJECT – USEFUL COMMANDS         ║
╚════════════════════════════════════════════════════╝

A practical cheat sheet for setting up and running a
Python-based LLM project on Linux (Ubuntu/Debian).

──────────────────────────────────────────────────────
SYSTEM SETUP
──────────────────────────────────────────────────────
Update system and install required tools:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip build-essential

```
──────────────────────────────────────────────────────
PROJECT SETUP
──────────────────────────────────────────────────────
Create project folder and enter it:
```bash
mkdir LLMproject
cd LLMproject

```
Clone a repository (optional):

```bash
git clone <repository_url>
cd <repository_name>
```

──────────────────────────────────────────────────────
PYTHON VIRTUAL ENVIRONMENT
──────────────────────────────────────────────────────
Create virtual environment:

```bash
python3 -m venv venv


Activate virtual environment:

source venv/bin/activate
```

Deactivate virtual environment:

```bash
deactivate

```
──────────────────────────────────────────────────────
DEPENDENCY MANAGEMENT
──────────────────────────────────────────────────────
Upgrade pip:

```bash
pip install --upgrade pip

```
Install dependencies:

```bash
pip install -r requirements.txt

```
Save installed packages:

```bash
pip freeze > requirements.txt
```

──────────────────────────────────────────────────────
RUNNING THE PROJECT
──────────────────────────────────────────────────────
Run a Python file:

```bash
python main.py

```


──────────────────────────────────────────────────────
FILE & FOLDER COMMANDS
──────────────────────────────────────────────────────
List files:

```bash
ls
ls -la

```
Create a file:

```bash
touch anewfile.py

```
View file contents:

```bash
cat anewfile.py
```

Edit file:

nano anewfile.py
vim anewfile.py


Remove a file:

rm file.py


Remove a folder:

rm -rf afolder


──────────────────────────────────────────────────────
GIT COMMANDS
──────────────────────────────────────────────────────
Use this section only if you want to use Git or learn Git.

git init
git status
git add .
git commit -m "Initial commit"
git pull
git push


──────────────────────────────────────────────────────
LLM / ML UTILITIES
──────────────────────────────────────────────────────

python --version
pip list
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"


──────────────────────────────────────────────────────
CLEANUP
──────────────────────────────────────────────────────

rm -rf venv


──────────────────────────────────────────────────────
NOTES
──────────────────────────────────────────────────────
• Always activate the virtual environment before running
