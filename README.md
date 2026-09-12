# Cuisync Web App

## Setup & Installation

Follow these steps to set up the environment and run the app locally.

### 1. Clone or Download the Repository
* Click the green **Code** button at the top of this repository page and select **Download ZIP**.
* Extract the `.zip` file to your computer.

### 2. Set Up a Virtual Environment (`venv`)
It is recommended to run this project inside a Python virtual environment to keep dependencies isolated. Type this into your terminal.

**On Windows:**
```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 3. Install Required Dependencies
With your virtual environment active, run:
```bash
pip install -r requirements.txt
```

### 4. Download Required Dataset Files
Download from this Drive folder: 
[Google Drive](https://drive.google.com/drive/folders/1OePoC482z8081cXCIbWCHIp_b-FbBUsh?usp=sharing)

and place all files in the main project folder alongside app.py.


### 5. Run program locally (for development only)
Run app.py and open it on your browser with this url:
```plaintext
http://127.0.0.1:8000
```

