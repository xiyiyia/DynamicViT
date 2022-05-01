import os

fo = open("ILSVRC2012_validation_ground_truth.txt", "r")

for label in fo:
    label_len = len(label)
    os.system('mv /home/ubuntu/datasets/ILSVRC2012_val/ILSVRC2012_val_000'
              + ''.join(['0' for i in range(5 - label_len)]) + label +
              '.JPEG')
fo.close()
