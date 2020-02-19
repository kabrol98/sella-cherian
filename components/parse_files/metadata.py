from os import stat


class FileMetaData:
    def __init__(self, file_path: str):
        self.file_path: str = file_path
        file_stats = stat(self.file_path)
        self.file_name: str = file_path.split("/")[-1]
        self.file_owner: int = file_stats.st_uid
        self.file_size: int = file_stats.st_size
        self.last_saved_user: str = None  # TODO: don't know; unavailable on Linux/macOS
        self.date_created: int = file_stats.st_ctime
        self.date_modified: int = file_stats.st_mtime

    def __str__(self):
        return f'''
file_path: {self.file_path},
file_name: {self.file_name},
file_owner: {self.file_owner},
file_size: {self.file_size}
date_created: {self.date_created}
date_modified: {self.date_modified}
'''


class ColumnMetaData:
    def __init__(self, file_name: str, sheet_name: str):
        self.file_name = file_name
        self.sheet_name = sheet_name

    def __str__(self):
        return f'''
file_name: {self.file_name},
sheet_name: {self.sheet_name},
'''
