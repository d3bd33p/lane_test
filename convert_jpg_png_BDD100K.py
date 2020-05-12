from PIL import Image
import sys
import os
import re



start_number = 0

def jpg2png(input_path, output_path):
    files = os.listdir(input_path)
    count = start_number
    for file in files:
        if not file.endswith('.jpg'):
                continue
        img = Image.open(input_path + file)
        img.save(output_path + file.split(".")[0] +'.png', 'png')
        count = count + 1




input_path = 'binary_lane_bdd/Images/'
output_path = 'binary_lane_bdd/Images/'
jpg2png(input_path, output_path)

input_path = 'binary_lane_bdd/Labels/'
output_path = 'binary_lane_bdd/Labels/'
jpg2png(input_path, output_path)



input_path = 'binary_lane_bdd/Images/'
files = os.listdir(input_path)
for file in files:
    print(file)



