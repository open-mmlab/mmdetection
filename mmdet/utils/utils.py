import os


def get_files(folder, file_format):
    files = []
    print("list number of files in each folders: ")
    for root, dirs, files in os.walk(folder):
        files = [f for f in files if file_format in f]
    return files
