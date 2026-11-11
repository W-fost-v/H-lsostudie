from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "health_study_data.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])