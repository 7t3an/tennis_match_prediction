Installation and Setup
1. Clone the repository
git clone <your-repository-URL>
cd tennis_match_prediction
2. Create and activate a virtual environment
python -m venv .venv
Activate the environment:

macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate
After activation, the terminal should show (.venv).
3. Install dependencies

Install general dependencies (latest compatible versions):
pip install -r requirements.txt
Install locked dependencies (exact versions that are guaranteed to work):
pip install -r requirements_locked.txt

requirements.txt is for development
requirements_locked.txt is for stable environment reproduction

Running the Project

Run the main Python script:
python main.py

Run the Streamlit app:
streamlit run app.py