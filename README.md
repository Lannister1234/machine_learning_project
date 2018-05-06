# machine_learning_project

This is an repository for 10701 machine learning course project

put extract_image_feature.py in the same directory of categories folders<br>
create a category.txt with categories which you want to be pretrained in it
and split them by line break.

command:<br>
* python extract_image_feature.py --model vgg16
* python read_features.py --model vgg16

| Files        | Descriptions |
| ------------- |:-------------:|
|classify_image.py      |     classify images|
|cnn_finetune_vgg16.py    |   fine-tune network|
|extract_image_feature.py |   extract features using vgg16 and so on|
|load_data.py            |    load data|
|read_features.py        |    read features from txt and train svm|
|resnet50_best.h5       |     weights after fine-tuning|
|result.xlsx           |      result|
|run.sh                |      for testing|