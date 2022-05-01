import os

fo = open("ILSVRC2012_validation_ground_truth.txt", "r")

for label in fo:
    label_len = len(label)
    os.system('mv /home/Ubuntu/datasets/ILSVRC2012/ILSVRC2012_val_000'
              + ''.join(['0' for i in range(5 - label_len)]) +
              '.JPEG')
fo.close()
