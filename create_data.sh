#!/bin/bash

# Check if directory exists and create if not
if [ ! -d "pix2tex/model/dataset" ]; then
    echo "Creating pix2tex/model/dataset directory..."
    mkdir -p pix2tex/model/dataset
else 
    echo "Dataset folder found! Moving to next steps..."
fi

# create tokenizer.jon
python -m pix2tex.dataset.dataset \
-e /space/hotel/vinhle/LaTex_Project/im2latex_100K-data/result/labels_train.txt \
/space/hotel/vinhle/LaTex_Project/im2latex_100K-data/result/labels_valid.txt \
/space/hotel/vinhle/LaTex_Project/im2latex_100K-data/result/labels_test.txt \
-s 1000 -o pix2tex/model/dataset/tokenizer.json  

# train data
echo "Creating Training data..."
python -m pix2tex.dataset.dataset \
-i /space/hotel/vinhle/LaTex_Project/im2latex_100K-data/result/train \
-e /space/hotel/vinhle/LaTex_Project/im2latex_100K-data/result/labels_train.txt \
-o pix2tex/model/dataset/train_100k.pkl

# valid data
echo " "
echo "Creating Validation data..."
python -m pix2tex.dataset.dataset \
-i /space/hotel/vinhle/LaTex_Project/im2latex_100K-data/result/valid \
-e /space/hotel/vinhle/LaTex_Project/im2latex_100K-data/result/labels_valid.txt \
-o pix2tex/model/dataset/valid_100k.pkl

# test data
echo " "
echo "Creating Test data..."
python -m pix2tex.dataset.dataset \
-i /space/hotel/vinhle/LaTex_Project/im2latex_100K-data/result/test \
-e /space/hotel/vinhle/LaTex_Project/im2latex_100K-data/result/labels_test.txt \
-o pix2tex/model/dataset/test_100k.pkl