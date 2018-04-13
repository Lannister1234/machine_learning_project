f = open("feature_and_labels.txt", 'r') 


labels = []
features = []
filenames = []  
for line in f.readlines():
    line = line.strip().split()
    labels.append(line[-2])
    filenames.append(line[-1])
    features.append(line[:-2])
f.close()