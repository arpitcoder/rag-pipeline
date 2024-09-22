# RAG Pipeline Project

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using document ingestion, embedding generation, and querying with OpenAI's GPT model.

## Prerequisites

- Python 3.12 or higher
- A macOS, Linux, or Windows machine
- OpenAI API key (for GPT model access)

## Setup and Run

### 1. Clone the Repository, Set Up Virtual Environment, and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline

# Create and activate a virtual environment

# On macOS or Linux:
python3 -m venv rag-env
source rag-env/bin/activate

# On Windows:
python -m venv rag-env
rag-env\Scripts\activate

# Install the required Python packages
pip install -r requirements.txt

# Configure OpenAI API Key

You can set your OpenAI API key either directly in the code or as an environment variable.

Option 1: Set API Key in Code
Edit querying.py and replace YOUR_API_KEY_HERE with your OpenAI API key:

# querying.py

`import openai`

# Set your OpenAI API key
`openai.api_key = "YOUR_API_KEY_HERE"`


# Option 2: Set API Key as Environment Variable

`export OPENAI_API_KEY="YOUR_API_KEY_HERE"`

# Run the Script

`python3 main.py`


## Troubleshooting
`API Key Error:` Ensure your OpenAI API key is correctly set in the code or as an environment variable.
`Dependencies:` If you encounter issues with package installations, verify that requirements.txt includes the correct package versions.

# License
This project is licensed under the MIT License. See the LICENSE file for details.