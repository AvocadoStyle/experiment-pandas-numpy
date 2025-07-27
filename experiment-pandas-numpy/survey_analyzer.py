import os
import pandas as pd
import numpy as np
CWD = os.path.join(os.getcwd(), 'experiment-pandas-numpy')
csv_path = os.path.join(CWD, 'db', 'survey_results_public.csv')
df = pd.read_csv(csv_path)

print("hello world")