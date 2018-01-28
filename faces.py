from pylab import *
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

actors = ["Alec Baldwin", "Bill Hader", "Steve Carell"]

actresses = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon"]

training_dictionary = {}
validation_dictionary = {}
test_dictionary = {}


def shuffle_images():
    for performer in act:
        if performer in actors:
            path = "cropped/actors"
        else:
            path = "cropped/actresses"
        actor_last_name = performer.split(" ")[1]
        flag = False
        image_list = []
        for filename in os.listdir(path):
            if actor_last_name.lower() in filename:
                flag = True
                image_list.append(filename)
            if actor_last_name.lower() not in filename and flag:
                break;
        # randomize and shuffle the images
        np.random.shuffle(image_list)
        counter = 0
        for image in image_list:
            if counter < 70:
                if actor_last_name.lower() in training_dictionary:
                    training_dictionary[actor_last_name.lower()].append(image)
                else:
                    training_dictionary[actor_last_name.lower()] = [image]
            if 70 <= counter < 80:
                if actor_last_name.lower() in validation_dictionary:
                    validation_dictionary[actor_last_name.lower()].append(image)
                else:
                    validation_dictionary[actor_last_name.lower()] = [image]
            if 80 <= counter < 90:
                if actor_last_name.lower() in test_dictionary:
                    test_dictionary[actor_last_name.lower()].append(image)
                else:
                    test_dictionary[actor_last_name.lower()] = [image]
            counter = counter + 1

def f(x, y, theta):
    """The cost function"""
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    """The Gradient"""
    return -2*sum((y-dot(theta.T, x))*x, 1)

def grad_descent(f, df, x, y, init_t, alpha, max_iter=10000):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 200 == 0:
            # print "Iter", iter
            # print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t))
            # print "Gradient: ", df(x, y, t), "\n"
            iter += 1
    return t

def hypothesis(theta, x_input):
    return dot(theta.T, x_input)

def get_performance(performer_1_full, performer_2_full, result_theta):
    performer_1 = performer_1_full.split(" ")[1].lower()
    performer_2 = performer_2_full.split(" ")[1].lower()
    if performer_1_full in actors:
        path_1 = "cropped/actors/"
    else:
        path_1 = "cropped/actresses/"

    if performer_2_full in actors:
        path_2 = "cropped/actors/"
    else:
        path_2 = "cropped/actresses/"

    # Check the performances of the classifier
    performer_1_validation_set = validation_dictionary[performer_1] # baldwin
    performer_2_validation_set = validation_dictionary[performer_2] # carell

    performer_1_test_set = test_dictionary[performer_1] # baldwin
    performer_2_test_set = test_dictionary[performer_2] # carell

    accurate_count_validation = 0

    for image in performer_1_validation_set:
        img = imread(path_1 + image)[:, :, 0]
        flatten_image = img.flatten() / 255.0
        result = hypothesis(result_theta, flatten_image)
        if result > 0.5:
            accurate_count_validation += 1

    for image in performer_2_validation_set:
        img = imread(path_2 + image)[:, :, 0]
        flatten_image = img.flatten() / 255.0
        result = hypothesis(result_theta, flatten_image)
        if result <= 0.5:
            accurate_count_validation += 1

    accuracy_validation = accurate_count_validation / float(len(performer_1_validation_set) + len(performer_2_validation_set))

    accurate_count_test = 0

    result_array = []

    for image in performer_1_test_set:
        img = imread(path_1 + image)[:, :, 0]
        flatten_image = img.flatten() / 255.0
        result = hypothesis(result_theta, flatten_image)
        result_array.append(result)
        if result > 0.5:
            accurate_count_test += 1

    # print "result array is ", result_array

    result_array_2 = []

    for image in performer_2_test_set:
        img = imread(path_2 + image)[:, :, 0]
        flatten_image = img.flatten() / 255.0
        result = hypothesis(result_theta, flatten_image)
        result_array_2.append(result)
        if result <= 0.5:
            accurate_count_test += 1

    # print "result_array_2 is ", result_array_2

    accuracy_test = accurate_count_test / float(len(performer_1_test_set) + len(performer_2_test_set))
    return accuracy_validation, accuracy_test


def binary_classify(performer_1_full, performer_2_full, training_samples_num=70, alpha=0.0000010, init_theta_coefficient=0):
    """Inputs are 2 actors' last names to classify
       Returns the trained thetas array for the hypothesis
    """
    performer_1 = performer_1_full.split(" ")[1].lower()
    performer_2 = performer_2_full.split(" ")[1].lower()
    if performer_1_full in actors:
        path_1 = "cropped/actors/"
    else:
        path_1 = "cropped/actresses/"

    if performer_2_full in actors:
        path_2 = "cropped/actors/"
    else:
        path_2 = "cropped/actresses/"
    # Training images for alec baldwin and steve Carel
    performer_1_training_set = training_dictionary[performer_1]
    performer_2_training_set = training_dictionary[performer_2]
    training_set = []
    performer_1_num_images = 0
    performer_2_num_images = 0

    for image_name in performer_1_training_set:
        if performer_1_num_images > training_samples_num:
            break
        path = path_1 + image_name
        image_file = imread(path)[:, :, 0]
        # Get the flatten image of inputs
        flatten_image = image_file.flatten()
        flatten_image_processed = flatten_image / 255.0  # so that each input is between 0 and 1
        training_set.append(flatten_image_processed) # training set 2D array
        performer_1_num_images = performer_1_num_images + 1

    for image_name in performer_2_training_set:
        if performer_1_num_images > training_samples_num:
            break
        path = path_2 + image_name
        image_file = imread(path)[:, :, 0]
        # Get the flatten image of inputs
        flatten_image = image_file.flatten()
        flatten_image_processed = flatten_image / 255.0  # so that each input is between 0 and 1
        training_set.append(flatten_image_processed) # training set 2D array
        performer_2_num_images = performer_2_num_images + 1

    x_matrix = np.vstack(training_set) # input x matrix for gradient descent. 70 rows = 70 images. 4096 columns = pixels
    # print "sum of x_matrix is:", sum(x_matrix)
    # print "init_theta_coefficient", init_theta_coefficient
    initial_theta = init_theta_coefficient * np.ones(len(x_matrix[0])) # initial thetas for graident descent
    # print "initial_theta", initial_theta
    # print "length of theta.T is ", len(initial_theta.T)
    # print "length of x_matrix is ", len(x_matrix)
    # print "length of x_matrix[0] is ", len(x_matrix[0])
    # print "theta.T length is ", initial_theta.T
    # print "x_matrix is ", x_matrix
    # print "performer_1_num_images is ", performer_1_num_images
    # print "performer_2_num_images is ", performer_2_num_images
    y_vector = np.ones(performer_1_num_images) # 1 for alec baldwin, 0 for Steve Carell
    y_vector = np.append(y_vector, np.zeros(performer_2_num_images))
    result_theta = grad_descent(f, df, x_matrix.T, y_vector, initial_theta, alpha)
    return result_theta

if __name__ == "__main__":

    # Split the images to training set, validation set and test set for part 2
    shuffle_images()

    # Create a classifier and performance scores for part 3
    result_theta = binary_classify("Alec Baldwin", "Steve Carell")
    validation_score, test_score = get_performance("Alec Baldwin", "Steve Carell", result_theta)
    print "Validation and test scores are respectively", validation_score, test_score

    # Display part 4 a)

    theta_image = np.reshape(result_theta, (32, 32))
    # print "img is ", img
    # print "img flatten is ", img.flatten()
    imsave("part_4a_full_training.jpg", theta_image)

    two_samples_theta = binary_classify("Alec Baldwin", "Steve Carell", 2)

    image_two_samples = np.reshape(two_samples_theta, (32, 32))

    imsave("part_4a_2_samples_training.jpg", image_two_samples)

    # part 4 b)

    new_starting_theta = binary_classify("Alec Baldwin", "Steve Carell", init_theta_coefficient=0.5)

    image_new_starting_theta = np.reshape(new_starting_theta, (32, 32))
    imsave("part_4b_new_start_theta.jpg", image_new_starting_theta)

    foggy_face_theta = binary_classify("Alec Baldwin", "Steve Carell")
    image_foggy_face_theta = np.reshape(foggy_face_theta, (32, 32))
    imsave("part_4b_foggy_face.jpg", image_foggy_face_theta)
