import numpy as np
import pandas as pd
import random

#writer = pd.ExcelWriter('demo.xlsx', engine='xlsxwriter')
#writer.save()

df_train = pd.read_csv('mnist_train.csv', header=None)
df_test = pd.read_csv('mnist_test.csv', header=None)
list_of_number_vectors = []

cm_row = ["|", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ""]
confusion_matrix = pd.DataFrame(np.array([cm_row, cm_row, cm_row, cm_row, cm_row, cm_row, cm_row, cm_row, cm_row, cm_row]),
                                 columns = ["|", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<- detected values"])


for i in range(10): #first nested list will store 0s, next will store 1s, etc.
    list_of_number_vectors.append([])

for i in range(len(df_train.index)):
    num = df_train.iloc[i, 0]
    list_of_number_vectors[num].append(df_train.iloc[i, df_train.columns!=0].to_numpy())


#sets all values in the confusion matrix to 0
def reset_confusion_matrix():
    for i in range(10):
        for j in range(10):
            confusion_matrix.iloc[i, j + 1] = 0


#k nearest neighbors
def test_100_numbers(k):
    reset_confusion_matrix()
    correct = 0
    incorrect = 0
    nearest_neighbors_lists = []
    for i in range(100):
        num_value = df_test.iloc[i, 0]
        num_vector = df_test.iloc[i, df_test.columns!=0].to_numpy()
        nn = k_neighbors_most_frequent(num_vector, k)
        confusion_matrix.iloc[num_value, nn + 1] = int(confusion_matrix.iloc[num_value, nn + 1]) + 1
        if num_value == nn:
            correct += 1
        else:
            incorrect += 1
    correct_percentage = correct / ((correct + incorrect) / 100)
    return str(correct) + " out of " + str(correct + incorrect) + " correct - " + str(correct_percentage) + "% accuracy."


#k nearest neighbors
def test_10000_numbers(k):
    reset_confusion_matrix()
    correct = 0
    incorrect = 0
    nearest_neighbors_lists = []
    for i in range(len(df_test.index)):
        num_value = df_test.iloc[i, 0]
        num_vector = df_test.iloc[i, df_test.columns!=0].to_numpy()
        nn = k_neighbors_most_frequent(num_vector, k)
        confusion_matrix.iloc[num_value, nn + 1] = int(confusion_matrix.iloc[num_value, nn + 1]) + 1
        if num_value == nn:
            correct += 1
        else:
            incorrect += 1
        if i % 100 == 0:
            print("progress: " + str(i) + " out of " + str(len(df_test.index)))
    correct_percentage = correct / ((correct + incorrect) / 100)
    return str(correct) + " out of " + str(correct + incorrect) + " correct - " + str(correct_percentage) + "% accuracy."


#used to find the 10 nearest neighbors in mnist_train
def find_10_nearest_neighbors(num):
    #used to set up nearest_neighbors variable - by default the first zero is the nearest neighbors
    nearest_neighbors = []
    neighbor = list_of_number_vectors[0][0]
    subtracted = np.subtract(num, neighbor)
    magnitude = np.linalg.norm(subtracted)
    nearest_neighbors.append([0, 0, magnitude])
    for i in range(len(list_of_number_vectors)):
        for j in range(len(list_of_number_vectors[i])):
            neighbor = list_of_number_vectors[i][j]
            subtracted = np.subtract(num, neighbor)
            magnitude = np.linalg.norm(subtracted)
            nn_magnitude = nearest_neighbors[-1][2]
            if magnitude < nn_magnitude: #if this neighbor is closer than the 10th closes neighbor
                for k in range(len(nearest_neighbors)):
                    if nearest_neighbors[k][2] > magnitude:
                        nearest_neighbors.insert(k, [i, j, magnitude])
                        break
                while len(nearest_neighbors) > 10:
                    nearest_neighbors.pop()
    return nearest_neighbors


def display_confusion_matrix():
    print(confusion_matrix)
    print("^inputs")



#returns the number that appears most frequently among k nearest neighbors
def k_neighbors_most_frequent(num, k):
    nearest_neighbors = find_10_nearest_neighbors(num)
    while len(nearest_neighbors) > k:
        nearest_neighbors.pop()
    neighbor_tally = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(k):
        neighbor = nearest_neighbors[i][0]
        neighbor_tally[neighbor] += 1
    most_frequent = 0
    for i in range(len(neighbor_tally)):
        if neighbor_tally[i] > neighbor_tally[most_frequent]:
            most_frequent = i
    return most_frequent



#used to print out a picture of an inputed number vector
def display_num(num):
    line = ""
    for i in range(28):
        for j in range(28):
            pixel = num[i*28 + j]
            if pixel == 0:
                #line = line + " "
                line += " "
            elif pixel < 100:
                line += "i"
            else:
                line += "M"
        print(line)
        line = ""
