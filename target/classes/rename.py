import os

folder = './train'

for i in range(len(os.listdir(folder))-1):
        dst = "train_" + str(i) + ".jar"
        src = f"{folder}/{os.listdir(folder)[i]}"
        dst =f"{folder}/{dst}"
        os.rename(src, dst)
