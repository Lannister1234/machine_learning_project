import sys
import os

'''
put this file in the same directory of classify_image.py, categories folders
create a category.txt with categories to be pretrained in it and split them
by line break.
'''
#parameters model
if len(sys.argv) != 2:
    print("cmd: python features_extract.py model.\ne.g. python features_extract.py vgg16.")
    exit()

model = sys.argv[1]

f = open("category.txt",'r') #
file = f.readlines()
f.close()
categories = []
for i in file:
    categories.append(i.strip())
    
    
f = open("feature_and_labels.txt", 'w')   
cmd = "python extract_image_feature.py --image %s --model %s"
for category in categories:
    for root, dirs, files in os.walk(category):
        for file in files:
            if file.endswith(".jpg"):
                try:
                    feature = os.popen(cmd % (file, model)).read().strip()
                    f.write(feature + ' ' + category + ' ' + file + '\n')  
                except:
                    pass
f.close()