import json
import shutil
from os import path, mkdir

from components.cell_labeling.cell_compact import CellTagType

tmp_folder_path = path.abspath(path.join(path.dirname(__file__), "../../tmp"))

def test_clean_up():
    try:
        shutil.rmtree(tmp_folder_path)
    except:
        pass

def test_exit():
    exit(1)

def test_dump(content, file_name):
    with open(path.join(tmp_folder_path, file_name), 'w') as f:
        f.write(json.dumps(content))

def get_tag_type_name(val):
    return CellTagType(val).name

class SimpleTest:
    def __enter__(self):
        try:
            test_clean_up()
            mkdir(tmp_folder_path)
        except:
            pass
    def __exit__(self):
        test_clean_up()

