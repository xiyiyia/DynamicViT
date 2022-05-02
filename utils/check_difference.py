import pickle

f1 = open("../result/correct_index0.5.pickle", "rb")
five = pickle.load(f1)
f2 = open("../result/correct_index0.7.pickle", "rb")
seven = pickle.load(f2)
count = 0
total = 0
for i in range(len(five)):
    for _, val in enumerate(five[i]):
        if val not in seven[i]:
            count += 1
        total += 1
print(count)
