import os

import shutil

mp4_folders_path = "/home/frb6002/Documents/lipread_files_mp4/trainval/"
text_folders_path = "/home/frb6002/Documents/lipread_files_wav/trainval"


for each_folder in os.listdir(mp4_folders_path):
    for each_file in os.listdir(os.path.join(mp4_folders_path, each_folder)):
        if each_file.endswith(".mp4"):
            text_file_name = each_file.replace(".mp4", ".txt")
            text_file_path = os.path.join(text_folders_path, each_folder, text_file_name)
            destination_file_path = os.path.join(mp4_folders_path, each_folder, text_file_name)
            shutil.copyfile(text_file_path, destination_file_path)

