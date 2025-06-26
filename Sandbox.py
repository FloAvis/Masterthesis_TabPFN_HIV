
files = [r"data/PI_DataSet.txt", r"data/INI_DataSet.txt", r"data/NRTI_DataSet.txt", r"data/NNRTI_DataSet.txt"]

for file in files:
    print(file)
    print(file.split("/")[-1].strip(".txt") + "_results.csv")