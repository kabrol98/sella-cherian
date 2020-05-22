# Sella
A large-scale relationships discovery tool for file-based tabular datasets.  

## branch: rework
The rework branch contains the codebase for the python-based sella system outlined in the paper submitted to the SSDBM conference in March 2020. It also contains a large corpus of Excel spreadsheets that can be used for further development and testing.

## BERT installation
In order to run Sella, you will need to successfully install and run the bert-as-service application found here: https://github.com/hanxiao/bert-as-service. More detailed instructions regarding the required environment can be found in BERT_INSTALL.md

## Testing: testing/pipeline_test.py
Run this python file as follows to generate a confusion matrix of column similarities across an excel spreadsheet.
