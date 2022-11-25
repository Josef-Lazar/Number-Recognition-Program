import numpy as np
import pandas as pd
import random

#writer = pd.ExcelWriter('demo.xlsx', engine='xlsxwriter')
#writer.save()

df_train = pd.read_csv('mnist_train.csv', header=None)
df_test = pd.read_csv('mnist_test.csv', header=None)
list_of_number_vectors = []



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



clusters = []
cluster_points = []
k = 20


def point_val(point):
    return point.iloc[0]

def point_vector(point):
    return point.drop(0, axis = 0).to_numpy()


#picks k random numbers from the training data set to act as cluster points
def generate_cluster_points():
    global clusters
    global cluster_points
    clusters = []
    cluster_points = []
    for i in range(k):
        cluster_point = df_train.iloc[random.randint(1, len(df_train.index) - 1), df_train.columns!=0].to_numpy()
        cluster_points.append(cluster_point)
        clusters.append([])


def clear_clusters():
    for i in range(len(clusters)):
        clusters[i] = []


def display_n_cluster_points(n):
    for i in range(min(k, n)):
        point = cluster_points[i]
        display_num(point)


def display_n_points_in_cluster(n, cluster):
    for i in range(min(n, len(cluster))):
        display_num(cluster[i])


def display_cluster(cluster):
    elements = ""
    for i in cluster:
        elements += str(point_val(i))
    print(elements)


def get_cluster_data(cluster):
    occurences = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in cluster:
        num = point_val(i)
        occurences[num] += 1
    return occurences


def display_cluster_data(cluster):
    occurences = get_cluster_data(cluster)
    print("number | occurences")
    for j in range(10):
        print("     " + str(j) + " | " + str(occurences[j]))


def display_distribution_matrix():
    occurences = []
    for i in range(k):
        occurences.append(["|"] + get_cluster_data(clusters[i]) + [""])
    distribution_matrix = pd.DataFrame(np.array([i for i in occurences]),
                                       columns = ["|", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<- numbers in cluster"])
    print(distribution_matrix)
    print("^clusters")

def sum_of_squared_errors():
    error_values = []
    for i in range(k):
        cluster_point = cluster_points[i]
        for point in clusters[i]:
            point_vec = point_vector(point)
            error = distance_between_vectors(cluster_point, point_vec)
            error_values.append(error)
    mean_of_errors = sum(error_values) / len(error_values)
    ev_minus_mean_squared = []
    for error in error_values:
        ev_minus_mean_squared.append((mean_of_errors - error) ** 2)
    return sum(ev_minus_mean_squared)
    #bassed on: https://sixsigmastudyguide.com/sum-of-squares-ss/


#internal metric
def average_distance_from_center():
    distance_sum = 0
    point_count = 0
    for i in range(k):
        center = cluster_points[i]
        for point in clusters[i]:
            point_vec = point_vector(point)
            distance_sum += distance_between_vectors(point_vec, center)
            point_count += 1
    return distance_sum / point_count


#external metric
def purity():
    purity_sum = 0
    for i in range(k):
        cluster_data = get_cluster_data(clusters[i])
        dominant_point_val = max(cluster_data)
        purity_sum += dominant_point_val / (dominant_point_val + sum(cluster_data))
    return purity_sum / k


#uses formula sqrt(x^2 + y^2) to calculate distance between two inputed points
def distance_between_vectors(vec1, vec2):
    distance = np.subtract(vec1, vec2)
    distance = np.square(distance)
    distance = np.sum(distance)
    distance = np.sqrt(distance)
    return distance


#returs index of cluster point that is closest to the inputed point
def closest_cluster_point(point):
    p_vec = point_vector(point)
    distance = distance_between_vectors(p_vec, cluster_points[0])
    closest = 0
    for i in range(1, k):
        new_distance = distance_between_vectors(p_vec, cluster_points[i])
        if new_distance < distance:
            distance = new_distance
            closest = i
    return closest


#compares the inputed point to the cluser points and assignes it to the cluster whose point it is closes to
def assign_point_to_cluster(point):
    cluster_point_index = closest_cluster_point(point)
    clusters[cluster_point_index].append(point)


#returns the point that is the mean of a cluster
def calculate_mean_of_cluster(cluster):
    sum_of_points = point_vector(cluster[0])
    for i in range(1, len(cluster)):
        sum_of_points = np.add(sum_of_points, point_vector(cluster[i]))
    mean = np.divide(sum_of_points, len(cluster))
    return mean


def cluster_100_points_n_times(n):
    global cluster_points
    generate_cluster_points()
    for x in range(n):
        clear_clusters()
        for i in range(100):
            num = df_train.iloc[i]
            assign_point_to_cluster(num)
        repetition = True
        for j in range(k):
            new_cluster_point = calculate_mean_of_cluster(clusters[j])
            if not np.array_equal(new_cluster_point, cluster_points[j]):
                repetition = False
            cluster_points[j] = new_cluster_point
        if repetition:
            print("repetition after " + str(x) + " iterations")
            break


def cluster_10000_points_n_times(n):
    global cluster_points
    generate_cluster_points()
    for x in range(n):
        print("progress: " + str(x) + " out of " + str(n))
        clear_clusters()
        for i in range(len(df_train.index)):
            num = df_train.iloc[i]
            assign_point_to_cluster(num)
        repetition = True
        for j in range(k):
            new_cluster_point = calculate_mean_of_cluster(clusters[j])
            if not np.array_equal(new_cluster_point, cluster_points[j]):
                repetition = False
            cluster_points[j] = new_cluster_point
        if repetition:
            print("repetition after " + str(x) + " iterations")
            break
