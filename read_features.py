f = open("feature_and_labels.txt", 'w') 


label = []
features = []
filename = []  
for line in f.readlines():
    line = line.strip().split()
    label.append(line[-2])
    filename.append(line[-1])
    features.append(line[:-2])
f.close()