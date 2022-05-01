import os

fo = open("ILSVRC2012_validation_ground_truth.txt", "r")
i = 1
for label in fo:
    label_len = len(label)
    index_len = len(str(i))
    os.system('mv /home/ubuntu/datasets/ILSVRC2012_val/ILSVRC2012_val_000'
              + ''.join(['0' for i in range(5 - index_len)]) + str(i) + '.JPEG'
              + ' /home/ubuntu/datasets/ILSVRC2012_val/val/'+label)
    i += 1
fo.close()
