import pandas as pd

folder = "~/LeandroSobralThesis/Yassef/"

parquet_file = folder + "peptides.parquet"
df = pd.read_parquet(parquet_file)
df.to_csv("peptides.csv")
