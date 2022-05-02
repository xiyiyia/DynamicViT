import pickle

f1 = open("../result/correct_index0.5.pickle", "rb")
five = pickle.load(f1)
f2 = open("../result/correct_index0.7.pickle", "rb")
seven = pickle.load(f2)
for i in range(len(five)):
    print(five[i] not in seven[i])
