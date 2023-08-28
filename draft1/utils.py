# utils.py
import shutil
import os

def delete_directory(path):
    try:
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    except Exception as e:
        print(f"Error deleting directory: {e}")

def create_empty_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Created empty directory: {path}")
    except Exception as e:
        print(f"Error creating directory: {e}")

def clean_directories():
    delete_directory("predictions/")
    create_empty_directory("predictions/")
    create_empty_directory("predictions/training")
    create_empty_directory("predictions/test")
    create_empty_directory("predictions/training/batch")
    create_empty_directory("predictions/test/batch")

    delete_directory("joblib")
    create_empty_directory("joblib")
