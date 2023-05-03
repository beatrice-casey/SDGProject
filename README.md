# SDG Project

## Installation instructions

1. Clone the code to your local repository.
2. Enter the main directory and clone graph2vec into this main directory from the following link: https://github.com/benedekrozemberczki/graph2vec/tree/master
3. You will most likely need two IDEs, one that supports Java and one that supports Python.
4. Navigate to a Java IDE and open the root folder. Run the following command:

    `mvn compile`

    If your environment is correct, this command should succeed.
5. Currently, the Java files in the reources folder are all the clean versions of the original code. If you want the original vulnerable code snippets, go to the following link and download the folder: https://github.com/find-sec-bugs/juliet-test-suite/tree/master/src/testcases/CWE15_External_Control_of_System_or_Configuration_Setting

6. All of the jar files have already been generated. However, if you want to compile new jar files, ensure you have any dependent files in the resources folder with the code you wish to compile, and run the following commands:
    `cd src/main/resources`

    `javac ClassName.java --release 8`

    `jar cvf train/train_#.jar *.class` 

    `rm *.class`

    The naming convention for the jar files are test_# where # is a number that follows the normal ordering convention starting from 0. E.g. if you have 3 jar files, then they should be named train_0.jar, train_1.jar and train_2.jar. Again, the train folder is already populated, but should you choose to have your own test cases please follow these instructions.

7. In Representations.java on line 34, there is a for loop that will iterate through every jar file and generate all four representations. Change the number of times the loop should execute based on the number of files you have. 

8. Run Representations.java. It will take quite some time to run but once it is complete, the different folders in the home directory named after each representation will be populated with the edgelists for each representation for every jar file. For example, the cfgs folder will contain cfg_0.edgelist, cfg_1.edgelist and so on.

9. Now transition to a Python editor and copy the edgelists from the aforementioned folders into the edgelists directories for each representation. Run main.py, again editing the range it iterates through for the number of jar files (and therefore edgelists) that you have.

10. Once this step is complete, you will have the json files needed to run graph2vec. Run the following command in your terminal from the PythonPortion directory:
    `python3 graph2vec/src/graph2vec.py --input-path graph2vec/dataset/graph2vec_input/representationName --output-path graph2vec/features/graphEmbedding_representationName.csv`

    Where the representation name is either cfg, pdg, sdg or cg. Run this command individually for each representation.

11. This has created the embeddings. For each representation you wish to run the ML model on, run the corresponding mlExperiments Python file. You may need to install some libraries, such as pandas, scikit-learn and pytorch. 

12. An image of the accuracy and loss plots will appear with the title of the representation followed by .png (eg. PDG.png) and the results for the experiments will print to terminal. 