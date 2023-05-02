import json


def convert_to_json(rep_type):

    for i in range(664):
        folder = rep_type + "_" + "edgelists"
        filename = rep_type + "_" + str(i) + ".edgelist"
        f = open(folder+'/'+filename, 'r')

        listEdges = []

        lines = f.readlines()

        for line in lines:
            line.replace("\n", "")
            edge = list(line.split(" "))
            if len(edge) >= 3:
                edge = [int(j) for j in edge[:2]]
                listEdges.append(edge[:2])

        edges = {'edges': listEdges}

        json_object = json.dumps(edges, indent=4)

        # Writing converted edgelist to .json for graph2vec input
        with open("graph2vec/dataset/graph2vec_input/" + rep_type + "/" + rep_type + "_" + str(i) +".json", "w") as outfile:
            outfile.write(json_object)


convert_to_json("cfg")
#convert_to_json("cg")
#convert_to_json("pdg")
#convert_to_json("sdg")


