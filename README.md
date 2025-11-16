# team4

## Setup

### Prerequisites (tested on)
- Python 3.11
- For GPU support (optional):
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit 12.2
  - cuDNN 8.9

### Installation

1. **Clone the repository**
```bash
   git clone 
   cd 
```

2. **Create a virtual environment**
   
   **Linux/macOS:**
```bash
   python3 -m venv env
   source env/bin/activate
```
   
   **Windows:**
```cmd
   python -m venv env
   env\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Verify GPU setup (optional)**
```bash
   python tf-gpu.py
```

### Platform-Specific Notes

#### Linux/WSL2
Tensorflow GPU support requires Linux/WSL2 with CUDA-enabled drivers (check prerequisites)

#### Windows (Native)
Tensorflow GPU support on native-Windows is only available for 2.10 or earlier versions

#### MacOS 12.0 or later
Tensorflow GPU support requires tensorflow-metal plugin, refer to the official installation guide

#### CPU-only (any operating system)
The project works on CPU without any additional setup, though training will be slower.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
