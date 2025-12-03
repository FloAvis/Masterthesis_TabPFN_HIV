# <ins>Multilabel prediction using tabular prior-data fitted networks applied to HIV drug resistance prediction</ins>

## Github Repository accompanying the Master Thesis in Bioinformatics of _Florian Benedikt Vögele_

## Description

The ongoing AIDS epidemic, caused by HIV, is still affecting millions of people worldwide. In recent decades, many advancements in antiviral drug development, especially research concerning HIV, have been made, resulting in a decreased disease burden in infected patients and a lower transmission rate. However, HIV drug resistance is on the rise, hindering treatment and increasing the risk of transmission. A correct treatment of HIV with effective drugs is vital for the management of the disease. Using the new machine learning model class prior-data fitted networks, this thesis aimed to predict HIV drug resistance based on mutations in the target proteins of the drugs. For this purpose, a multi-label prediction approach was used to leverage multiple drugs targeting the same proteins and overlapping mutation predictors across different drugs. \\
in this thesis, it has been shown that HIV drug resistance is predictable by simple multi-label classification methods and that additional information about other drug resistances can improve the predictive power. Additionally, modifying prior-data fitted networks for the inherent prediction of multi-label data is possible to a degree, but needs further research to confirm its full potential.

The repository contains the code used for the dataset preprocessing, preliminary analyses and PT prediction approaches.

The repositroy containing the code for the retraining of TabPFN and TabICL for model adaption can be found [here](https://github.com/FloAvis/MultiLabel-PFN/).

## Getting Started

### Dependencies

Dependencies are described in the requirenments.txt file and a more conclusive list of packages used during the Thesis can be found under thesis_package_list.txt

### Repository structure

* Data
  * datasets are saved under data/
  * The HIVDRD are saved as their high quality filtered and the complete datasets with the suffix of .Full
  * The benchmarking datasets are saved under data/Other_dataset as .arff and processed .csv files


* Predictions
  * scripts for multi-label predictions can be found under predictions/
  * The scripts load the HIVDRD and either perform a single prediction absed on a training and test data split or perform k-fold cross validation
  * The predicted labels as well as well as the probabilities are saved in prediction_results/


* Statistics
  * the script for the evaluation of the predictions can be found under statistics/
  * It consists of jupyter notebooks with the evaluations


* Helper functions:
  * data_preprocessing: provides functions for data preprocessing
  * prediction_handler: provides functions for cross validation and predictions
  * result_handler: provides functions for saving prediction results and metric calculation
  * Classifiers: provides BR and CC as well as function for CC ensemble

## Help

The applications of the functions are described at the functions itself. If more questions arise you can contact me under florian.voegele@hotmail.de.

## Author

Florian Benedikt Vögele  


## Version History

* 0.0.1
  * Hand in of Thesis

## License

MIT License

Copyright (c) 2025 Florian Benedikt Vögele

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



