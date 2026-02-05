#  Python LLM Project – Useful Commands

---

##  System Setup

### Update the system and install required tools:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip build-essential
```
## Project Setup

### Create a project folder and enter it:

```bash
mkdir LLMproject
cd LLMproject
```
### Clone a repository (optional):

```bash
git clone <repository_url>
cd <repository_name>
```
## Python Virtual Environment

### Create a virtual environment:

```bash
python3 -m venv venv

```

### Activate the virtual environment:

```bash
source venv/bin/activate

```
### Deactivate the virtual environment:

```bash
deactivate
```

## Dependency Management

### Upgrade pip:

```bash
pip install --upgrade pip

```

### Install dependencies:

```bash
pip install -r requirements.txt
```
###Save installed packages:

```bash
pip freeze > requirements.txt
```
## Running the Project

### Run a Python file:
```bash
python main.py
```
### or
```bash
python3 main.py
```

# File & Folder Commands

## List files:
```bash
ls
ls -la
```
##Create a file:

```bash
touch anewfile.py
```
### View file contents:

```bash
cat anewfile.py
```

## Edit a file:

```bash
nano anewfile.py
# or
vim anewfile.py
```
##Remove a file:
```bash
rm file.py
```

## Remove a folder:
```bash
rm -rf afolder
```
# Git Commands

## Use this section if you want to use or learn Git:
```bash
git init
git status
git add .
git commit -m "Initial commit"
git pull
git push
```
# LLM / ML Utilities

## Useful checks for Python, packages, and GPU support:

```bash
python --version
pip list
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```
# Cleanup

## Remove the virtual environment:
```bash
rm -rf venv
```
