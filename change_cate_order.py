f = open("feature_and_labels_resnet.txt", 'r')
labels = []
features = []
filenames = []

k = open("feature_and_labels_resnet_t.txt", 'w')

l = ['Fear','Happiness','Passion','Peaceful','Sadness']
cate = {'Fear':[],'Happiness':[],'Passion':[],'Peaceful':[],'Sadness':[]}

for line in f.readlines():
    line_t = line.strip().split()
    label = line_t[-2]
    cate[label].append(line)
    
for i in l:
    for line in cate[i]:
        k.write(line)

f.close()
k.close()