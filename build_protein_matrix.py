import pandas as pd

folder = "~/LeandroSobralThesis/Yassef/"
tsv_file = folder + "ibaq.absolute.tsv"
data = pd.read_csv(
    tsv_file,
    sep="\t",
    lineterminator="\n",
    skiprows=(10),
    header=(0),
    usecols=(0, 1, 4),
)
data.to_csv(folder + "ibaq.absolute.csv", index=False)

print(data)

matrix = data.pivot(index="protein", columns="sample_accession", values="ribaq")

print(matrix)

matrix.to_csv(folder + "human_tissue.csv")

print(matrix.isna().sum().sum() / matrix.size * 100)
