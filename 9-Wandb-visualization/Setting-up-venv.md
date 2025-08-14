## Installation Guide: ffmpeg & Python Virtual Environment

### 0. Make sure you're in the right folder

```bash
cd 9-Wandb-visualization
```

### 1. Install ffmpeg

Ubuntu/Debian:

```bash
sudo apt update && sudo apt install ffmpeg
```

RHEL/CentOS (EPEL required):

```bash
sudo dnf install epel-release &&
sudo dnf install ffmpeg
```

### 2. Set up Python venv (visualization_env)

```bash
python3 -m venv visualization_env &&
source visualization_env/bin/activate &&
pip install --upgrade pip &&
pip install -r requirements.txt
```

### 3. Make sure to use the virtual environment when running the scripts

To activate later:

```bash
source visualization_env/bin/activate
```
