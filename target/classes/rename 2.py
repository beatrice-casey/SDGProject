import os

folder = './train'
for count, filename in enumerate(os.listdir(folder)):
        dst = f"train_{str(count)}.jar"
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)
