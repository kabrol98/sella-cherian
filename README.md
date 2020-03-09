# sella
A large-scale relationships discovery tool for file-based tabular datasets.  

## branch: rework
A revised system of previous codebase

## develop and build
Recommend using conda for package managing
* Use `conda list --export > package-list.txt` to freeze requirements
* Use `conda create -n env --file package-list.txt` to recreate virtual environment
* Use `conda activate ./env/` to activate environment
* Use `conda deactivate` to exit environment

## Testing: testing/pipeline_test.py
Run this python file as follows to generate a confusion matrix of column similarities across an excel spreadsheet.

Usage: