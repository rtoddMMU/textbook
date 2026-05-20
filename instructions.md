# Building and Viewing the Textbook

## Prerequisites

- Anaconda or Miniconda installed
- Node.js (will be installed automatically by jupyter-book if needed)

## Setup Environment

```bash
# Create conda environment
conda create -n data8-pandas python=3.11 -y

# Activate environment
conda activate data8-pandas

# Install dependencies
pip install jupyter-book pandas numpy matplotlib scipy

# If there's a requirements.txt, also run:
pip install -r requirements.txt

# From the textbook root directory
myst build

# Navigate to the build output----this didn't work for me
cd _build/site

# Start the server
python -m http.server 8000

# From the textbook root directory --- this did work for me
myst start

# 1. Edit a notebook or markdown file
# (make your changes in Jupyter, VS Code, etc.)

# 2. Rebuild
myst build

# 3. Refresh your browser
# The server keeps running, just reload the page