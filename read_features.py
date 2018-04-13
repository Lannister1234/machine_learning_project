f = open("feature_and_labels.txt", 'r') 


labels = []
features = []
filenames = []  
for line in f.readlines():
    line = line.strip().split()
    labels.append(line[-2])
    filenames.append(line[-1])
    temp = [float(i) for i in line[:-2]]
    features.append(temp)
f.close()