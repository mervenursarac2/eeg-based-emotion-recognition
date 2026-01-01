import numpy as np
import pandas as pd


df = pd.read_excel("C:\\Users\\merve\\Desktop\\deap-dataset\\metadata_xls\\participant_ratings.xls")

df['Arousal_Label'] = np.where(df['Arousal'] > 5.0, 1, 0)
df['Valence_Label'] = np.where(df['Valence'] > 5.0, 1, 0)

print("labeled the data based on a threshold of 5.0.")

print(df.head())