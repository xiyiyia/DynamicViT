import pickle

# f1 = open("../result/correct_index0.5.pickle", "rb")
# five = pickle.load(f1)
# f2 = open("../result/correct_index0.7.pickle", "rb")
# seven = pickle.load(f2)
# count = 0
# total = 0
# for i in range(len(five)):
#     for _, val in enumerate(five[i]):
#         if val not in seven[i]:
#             count += 1
#         total += 1
# print(count, total)
# f1.close()
# f2.close()
#
# f1 = open("../result/correct_index0.7.pickle", "rb")
# five = pickle.load(f1)
# f2 = open("../result/correct_index0.5.pickle", "rb")
# seven = pickle.load(f2)
# count = 0
# total = 0
# for i in range(len(five)):
#     for _, val in enumerate(five[i]):
#         if val not in seven[i]:
#             count += 1
#         total += 1
# print(count,total)
# f1.close()
# f2.close()
#
# f1 = open("../result/correct_index0.6.pickle", "rb")
# five = pickle.load(f1)
# f2 = open("../result/correct_index0.7.pickle", "rb")
# seven = pickle.load(f2)
# count = 0
# total = 0
# for i in range(len(five)):
#     for _, val in enumerate(five[i]):
#         if val not in seven[i]:
#             count += 1
#         total += 1
# print(count,total)
# f1.close()
# f2.close()
#
# f1 = open("../result/correct_index0.5.pickle", "rb")
# five = pickle.load(f1)
# f2 = open("../result/correct_index0.6.pickle", "rb")
# seven = pickle.load(f2)
# count = 0
# total = 0
# for i in range(len(five)):
#     for _, val in enumerate(five[i]):
#         if val not in seven[i]:
#             count += 1
#         total += 1
# print(count,total)
# f1.close()
# f2.close()

f1 = open("../result/correct_index0.5.pickle", "rb")
five = pickle.load(f1)
f2 = open("../result/correct_index0.7.pickle", "rb")
seven = pickle.load(f2)
f3 = open("../result/correct_index0.6.pickle", "rb")
six = pickle.load(f3)

count = 0
total = 0
for i in range(len(five)):
    for _, val in enumerate(five[i]):
        if val not in seven[i]:
            count += 1
        total += 1
print(count, total)

count = 0
total = 0
for i in range(len(five)):
    for _, val in enumerate(five[i]):
        if val not in six[i]:
            count += 1
        total += 1
print(count, total)

count = 0
total = 0
for i in range(len(six)):
    for _, val in enumerate(six[i]):
        if val not in seven[i]:
            count += 1
        total += 1
print(count, total)

count = 0
total = 0
for i in range(len(five)):
    for _, val in enumerate(five[i]):
        if val in seven[i] and val in six[i]:
            count += 1
        total += 1
print(count, total)

f1.close()
f2.close()