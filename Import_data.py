#File for importing the stanford database datasets, high quality and full

from urllib.request import urlretrieve


#corresponding urls
urls = ["https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/PI_DataSet.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/NRTI_DataSet.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/NNRTI_DataSet.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/INI_DataSet.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/PI_DataSet.Full.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/NRTI_DataSet.Full.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/NNRTI_DataSet.Full.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/INI_DataSet.Full.txt"]


#output directory
data_storage = r".\data"

#importing files and saving them
for url in urls:
    filename = url.split("/")[-1]

    urlretrieve(url, data_storage + "/" + filename)

