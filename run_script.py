import os
import argparse

def execute_sh_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".sh"):
                sh_file_path = os.path.join(root, file)
                print(f"Executing {sh_file_path}")
                os.system(f"chmod +x {sh_file_path}")  # Ensure the script is executable
                os.system(sh_file_path)
def main():
    execute_sh_files_in_directory('./scripts')

if __name__ == "__main__":
    main()
