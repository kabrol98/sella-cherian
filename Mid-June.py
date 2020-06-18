import pickle

filename = "etl/tmp/etl_clustering_10.0_2020-06-18.pickle"
infile = open(filename,'rb')
true_results = pickle.load(infile)
infile.close()

real_results = {}

#change this to the relation you are testing
file = open("training_data/row_containment/spreadsheet_relationships.csv", "r")
for line in file.readlines():
    components = line.split(", ")
    #key is the original spreadsheet, value is the derived spreadsheet.
    key = components[0] + "." + components[1]
    value = components[2] + "." + components[3]
    #I can do this because for every spreadsheet currently there is only one derived spreadsheet
    real_results[key] = value

compared_final = set()
for group in true_results['text_names']:
    for i in range(len(group)):
        key = ".".join(group[i].split("/")[-1].split(".")[:-1])
        if key not in real_results:
            continue
        for j in range(len(group)):
            value = ".".join(group[j].split("/")[-1].split(".")[:-1])
            if real_results[key] == value:
                print(key, value)
                compared_final.add((key, value))

for group in true_results['numeric_names']:
    for i in range(len(group)):
        key = ".".join(group[i].split("/")[-1].split(".")[:-1])
        if key not in real_results:
            continue
        for j in range(len(group)):
            value = ".".join(group[j].split("/")[-1].split(".")[:-1])
            if real_results[key] == value:
                compared_final.add((key, value))

true_positive_rate = len(compared_final) / len(real_results)
print(true_positive_rate)
