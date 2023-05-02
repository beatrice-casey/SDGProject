# import required module
import os
# assign directory
directory = 'resources'
save_path = 'train'

# iterate over files in
# that directory

for filename in range(len(os.listdir('.'))):
    #f = os.path.join(directory, os.listdir('.')[filename])
    # checking if it is a file
    f = os.listdir('.')[filename]
    #print("javac " + f + " --release 8")
    os.system("javac " + f + " --release 8")
    if os.path.isfile(f) and f.endswith(".class"):
        f_name = "train_" + str(filename) + ".jar"
        save = os.path.join(save_path, f_name)
        command = "jar cvf resources/train/" +  f_name + " resources/*.class" 
        print(command)
        #os.system(command)

