# sella
A large-scale relationships discovery tool for file-based tabular datasets.  

## branch: rework
A revised system of previous codebase

## develop and build
Recommend using conda for package managing
* Use `conda list --export > package-list.txt` to freeze requirements
* Use `conda create -n env --file package-list.txt` to recreate virtual environment
* Use `conda activate ./env/` to activate environment
* Use 'conda install python=3.6' since we used tensorflow package
* Use 'python -m pip install --upgrade pip' to ensure pip is compatible with python
* Use 'pip install --user --requirement requirements.txt' to install all libraries
* Use `conda deactivate` to exit environment

## Testing: testing/pipeline_test.py
Run this python file as follows to generate a confusion matrix of column similarities across an excel spreadsheet.

Usage: ```
python testing/pipeline_test.py --help                    
usage: pipeline_test.py [-h] [-f FILENAME] (-s | -e) (-n | -t)

Tests sella pipeline on given excel spreadsheet. Outputs Confusion matrix of
column similarity test into testing/confusion_results. Use command line
arguments to configure different test types.

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Specify Excel spreadsheet name in data_corpus
                        directory (Omit .xlsx)
  -s, --standard        Use standard column summaries.
  -s, --extended        Use extended column summaries.
  -d, --numeric         Run tests on numeric columns
  -d, --text            Run tests on text columns
  ```
