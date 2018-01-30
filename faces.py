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

######################## IMPORTANT! PLEASE READ! ##################################

# Due to the fact that imsave from the pylab was used in get_data.py, the cropped
# images I got was of the shape (32, 32, 4) instead of (32, 32). As a result,
# when these images are opened using imread I got 4096 pixels instead of 1024 (32x32) due to
# the extra channel the cropped images has. To remedy the situation, whenever I
# read the cropped images, I always slice the images through [:, :, 0] in order to
# get rid of the extra channel and read the images as 32x32 for the purpose of this assignment.

##################################################################################


act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

actors = ["Alec Baldwin", "Bill Hader", "Steve Carell"]

actresses = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon"]

actors_different = ['Gerard Butler', 'Michael Vartan', 'Daniel Radcliffe']
actresses_different = ['America Ferrera', 'Kristin Chenoweth', 'Fran Drescher']
act_different = ['Gerard Butler', 'Michael Vartan', 'Daniel Radcliffe', 'America Ferrera', 'Kristin Chenoweth', 'Fran Drescher']
different_performer_dictionary = {}


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

def set_up_images_for_different_performers():
    for performer in act_different:
        if performer in actors_different:
            path = "cropped/actors"
        else:
            path = "cropped/actresses"
        actor_last_name = performer.split(" ")[1].lower()
        flag = False
        image_list = []
        for filename in os.listdir(path):
            if actor_last_name in filename:
                flag = True
                image_list.append(filename)
            if actor_last_name not in filename and flag:
                break;

        for image in image_list:
            if actor_last_name in different_performer_dictionary:
                different_performer_dictionary[actor_last_name].append(image)
            else:
                different_performer_dictionary[actor_last_name] = [image]


def f(x, y, theta):
    """The cost function"""
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    """The Gradient"""
    print "x is ", x
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

def get_performance_binary(performer_1_full, performer_2_full, result_theta):
    """Returns the accuracy of the training sets and validation sets"""
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

    performer_1_training_set = training_dictionary[performer_1] # baldwin
    performer_2_training_set = training_dictionary[performer_2] # carell

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

    accurate_count_training = 0

    result_array = []

    for image in performer_1_training_set:
        img = imread(path_1 + image)[:, :, 0]
        flatten_image = img.flatten() / 255.0
        result = hypothesis(result_theta, flatten_image)
        result_array.append(result)
        if result > 0.5:
            accurate_count_training += 1

    # print "result array is ", result_array

    result_array_2 = []

    for image in performer_2_training_set:
        img = imread(path_2 + image)[:, :, 0]
        flatten_image = img.flatten() / 255.0
        result = hypothesis(result_theta, flatten_image)
        result_array_2.append(result)
        if result <= 0.5:
            accurate_count_training += 1

    # print "result_array_2 is ", result_array_2

    accuracy_training = accurate_count_training / float(len(performer_1_training_set) + len(performer_2_training_set))
    return accuracy_validation, accuracy_training


def binary_classify(performer_1_full, performer_2_full, training_samples_num=70, alpha=0.0000010, init_theta_coefficient=0):
    """Inputs are 2 actors' last names to classify
       Returns the trained thetas array for the hypothesis and the cost function values
       for the training and validation sets
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
    performer_1_validation_set = validation_dictionary[performer_1]
    performer_2_validation_set = validation_dictionary[performer_2]
    training_set = []
    validation_set = []
    performer_1_num_images = 0
    performer_2_num_images = 0
    performer_1_num_images_validation = 0
    performer_2_num_images_validation = 0

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

    for image_name in performer_1_validation_set:
        if performer_1_num_images_validation > training_samples_num:
            break
        path = path_1 + image_name
        image_file = imread(path)[:, :, 0]
        # Get the flatten image of inputs
        flatten_image = image_file.flatten()
        flatten_image_processed = flatten_image / 255.0  # so that each input is between 0 and 1
        validation_set.append(flatten_image_processed) # training set 2D array
        performer_1_num_images_validation = performer_1_num_images_validation + 1

    for image_name in performer_2_validation_set:
        if performer_2_num_images_validation > training_samples_num:
            break
        path = path_1 + image_name
        image_file = imread(path)[:, :, 0]
        # Get the flatten image of inputs
        flatten_image = image_file.flatten()
        flatten_image_processed = flatten_image / 255.0  # so that each input is between 0 and 1
        validation_set.append(flatten_image_processed) # training set 2D array
        performer_2_num_images_validation = performer_2_num_images_validation + 1

    x_matrix = np.vstack(training_set) # input x matrix for gradient descent. 70 rows = 70 images. 4096 columns = pixels
    x_matrix_validation = np.vstack(validation_set)
    # print "sum of x_matrix is:", sum(x_matrix)
    # print "init_theta_coefficient", init_theta_coefficient
    initial_theta = init_theta_coefficient * np.ones(len(x_matrix[0])) # initial thetas for graident descent
    y_vector = np.ones(performer_1_num_images) # 1 for alec baldwin, 0 for Steve Carell
    y_vector = np.append(y_vector, np.zeros(performer_2_num_images))

    y_vector_validation = np.append(np.ones(performer_1_num_images_validation), np.zeros(performer_2_num_images_validation))
    result_theta = grad_descent(f, df, x_matrix.T, y_vector, initial_theta, alpha)
    result_cost_training = f(x_matrix.T, y_vector, result_theta)
    result_cost_validation = f(x_matrix_validation.T, y_vector_validation, result_theta)
    return result_theta, result_cost_training, result_cost_validation

def classify_gender(actors, actresses, act, training_samples_num=70, alpha=0.0000010, init_theta_coefficient=0):
    """Classify gender using training sets of performers in act
       Training labels:
       Male = 1
       Female = 0
    """
    training_set = []
    y_vector_list = []
    for performer in act:
        if performer in actors:
            path = "cropped/actors/"
            label_value = 1
        else:
            path = "cropped/actresses/"
            label_value = 0
        performer_last = performer.split(" ")[1].lower()
        # Set of images for the performer
        images = training_dictionary[performer_last]
        num_images = 0
        for image_name in images:
            if num_images > training_samples_num:
                break
            path_image = path + image_name
            image_file = imread(path_image)[:, :, 0]
            # Get the flatten image of inputs
            flatten_image = image_file.flatten()
            flatten_image_processed = flatten_image / 255.0  # so that each input is between 0 and 1
            training_set.append(flatten_image_processed) # training set 2D array
            num_images += 1
        if label_value == 1:
            label_array = np.ones(num_images)
        else:
            label_array = np.zeros(num_images)
        y_vector_list.append(label_array)

    x_matrix = np.vstack(training_set)
    initial_theta = init_theta_coefficient * np.ones(len(x_matrix[0]))
    # Construct the y label vector
    y_vector = np.array(y_vector_list)
    y_vector = y_vector.flatten()
    result_theta = grad_descent(f, df, x_matrix.T, y_vector, initial_theta, alpha)
    return result_theta

def get_performance_gender(actors, actresses, act, act_different, actors_different, actresses_different, result_theta):
    """Returns the accuracy of the training sets and validation sets"""
    accurate_count_validation = 0
    accurate_count_training = 0
    image_number_validation = 0
    image_number_training = 0

    different_six_performer_results = {}

    total_performers = act + act_different

    for performer in total_performers:
        if performer in actors or performer in actors_different:
            path = "cropped/actors/"
        else:
            path = "cropped/actresses/"
        threshold = 0.5 # > 0.5 implies Male, < 0.5 implies female. y = 1 <==> Male; y = 0 <==> Female
        performer_last_name = performer.split(" ")[1].lower()

        if performer in act:
            performer_validation_set = validation_dictionary[performer_last_name]
            performer_training_set = training_dictionary[performer_last_name]

            # validation set accuracy
            for image in performer_validation_set:
                path_image = path + image
                img = imread(path_image)[:, :, 0]
                flatten_image = img.flatten() / 255.0
                result = hypothesis(result_theta, flatten_image)
                if performer in actors:
                    if result > threshold:
                        accurate_count_validation += 1
                else:
                    if result <= threshold:
                        accurate_count_validation += 1
                image_number_validation += 1

            # training set accuracy
            for image in performer_training_set:
                path_image = path + image
                img = imread(path_image)[:, :, 0]
                flatten_image = img.flatten() / 255.0
                result = hypothesis(result_theta, flatten_image)
                if performer in actors:
                    if result > threshold:
                        accurate_count_training += 1
                else:
                    if result <= threshold:
                        accurate_count_training += 1
                image_number_training += 1
        else:
            different_performer_images = different_performer_dictionary[performer_last_name]
            different_performer_accurate_acount = 0
            different_image_count = 0

            for image in different_performer_images:
                path_image = path + image
                img = imread(path_image)[:, :, 0]
                flatten_image = img.flatten() / 255.0
                result = hypothesis(result_theta, flatten_image)
                if performer in actors_different:
                    if result > threshold:
                        different_performer_accurate_acount += 1
                else:
                    if result <= threshold:
                        different_performer_accurate_acount += 1
                different_image_count += 1

            accuracy_different_performer = different_performer_accurate_acount / float(different_image_count)
            different_six_performer_results[performer] = accuracy_different_performer

    accuracy_validation = accurate_count_validation / float(image_number_validation)
    accuracy_training = accurate_count_training / float(image_number_training)

    return accuracy_validation, accuracy_training, different_six_performer_results

def plot_performance_gender(actors, actresses, act, act_different, actors_different, actresses_different):
    """Plots the performance of validation and training sets vs training set size"""
    # Goes in steps of 4
    x_axis_sizes = [i for i in range(2, 70, 4)]
    performance_validation_set = []
    performance_training_set = []

    for i in x_axis_sizes:
        result_theta = classify_gender(actors, actresses, act, training_samples_num=i)
        validation_score, training_score, different_six_performer_results = get_performance_gender(actors, actresses, act, act_different, actors_different, actresses_different, result_theta)
        print "validation_score, training score are", validation_score, training_score
        performance_validation_set.append(validation_score)
        performance_training_set.append(training_score)

    figure_validation = plt.figure()
    plt.plot(x_axis_sizes, performance_validation_set, 'r-', label="Validation Set Accuracy")
    plt.plot(x_axis_sizes, performance_training_set, label="Training Set Accuracy")
    plt.xlabel("Training set size")
    plt.ylabel("Percent Accuracies")
    plt.title("Performance vs Size of Training Set")
    plt.legend(loc="best")
    plt.savefig("part_5_validation_training_plot")

def f_multiclass(x, y, theta):
    """The multiclass cost function"""
    return sum( (y - dot(theta.T,x)) ** 2)

def df_multiclass(x, y, theta):
    """The multiclass Gradient"""
    return -2*(y-dot(theta.T, x))*x

def part_6_finite_difference(f_multiclass, df_multiclass, k=4, init_theta_coefficient=0, alpha=0.0000010):
    training_set = []
    # Choose 5 images
    path = "cropped/actors/"

    for i in range(5):
        image_name = training_dictionary["carell"][i]
        path_image = path + image_name
        image_file = imread(path_image)[:, :, 0]
        # Get the flatten image of inputs
        flatten_image = image_file.flatten()
        flatten_image_processed = flatten_image / 255.0
        training_set.append(flatten_image_processed)

    x_matrix = np.vstack(training_set)
    initial_theta = []
    for i in range(k):
        initial_theta_row = init_theta_coefficient * np.ones(len(x_matrix[0]))
        initial_theta.append(initial_theta_row)
    initial_theta = np.vstack(initial_theta)
    y_vector = np.array([[0, 1, 0, 0] for i in range(5)])
    return grad_descent(f_multiclass, df_multiclass, x_matrix, y_vector, initial_theta, alpha, max_iter=10000)


if __name__ == "__main__":

    np.random.seed(0)

    ## Split the images to training set, validation set and training set for part 2
    shuffle_images()

    ## Set up the different_performer_dictionary for performers not in act
    set_up_images_for_different_performers()

    # ## Part 3 - Create a classifier and report performance scores
    # result_theta, result_cost_training, result_cost_validation = binary_classify("Alec Baldwin", "Steve Carell")
    # validation_score, training_score = get_performance_binary("Alec Baldwin", "Steve Carell", result_theta)
    #
    # print "Binary Classification Validation and training scores are respectively", validation_score, training_score
    # print "result_cost_training, result_cost_validation", result_cost_training, result_cost_validation
    # part_3_data_file = open("part_3_data_file.txt", "w")
    # part_3_data_file.write("The cost function value for the training set is " + str(result_cost_training) + "\n")
    # part_3_data_file.write("The cost function value for the validation set is " + str(result_cost_validation) + "\n")
    # part_3_data_file.write("The performance of the classifier (percent accuracy) on the training set is " + str(training_score) + "\n")
    # part_3_data_file.write("The performance of the classifier (percent accuracy) on the validation set is " + str(validation_score) + "\n")
    # part_3_data_file.close()
    #
    # ## Display part 4 a)
    #
    # theta_image = np.reshape(result_theta, (32, 32))
    # ## print "img is ", img
    # ## print "img flatten is ", img.flatten()
    # imsave("part_4a_full_training.jpg", theta_image)
    #
    # two_samples_theta, result_cost_training, result_cost_validation = binary_classify("Alec Baldwin", "Steve Carell", 2)
    #
    # image_two_samples = np.reshape(two_samples_theta, (32, 32))
    #
    # imsave("part_4a_2_samples_training.jpg", image_two_samples)
    #
    # ## part 4 b)
    #
    # new_starting_theta, result_cost_training, result_cost_validation = binary_classify("Alec Baldwin", "Steve Carell", init_theta_coefficient=0.5)
    #
    # image_new_starting_theta = np.reshape(new_starting_theta, (32, 32))
    # imsave("part_4b_new_start_theta.jpg", image_new_starting_theta)
    #
    # foggy_face_theta, result_cost_training, result_cost_validation = binary_classify("Alec Baldwin", "Steve Carell")
    # image_foggy_face_theta = np.reshape(foggy_face_theta, (32, 32))
    # imsave("part_4b_foggy_face.jpg", image_foggy_face_theta)
    #
    # ## part 5
    # part_5_result_theta = classify_gender(actors, actresses, act)
    # part_5_validation_score, part_5_training_score, different_six_performer_results = get_performance_gender(actors, actresses, act, act_different, actors_different, actresses_different, part_5_result_theta)
    # print "part 5 validation and training scores are respectively ", part_5_validation_score, part_5_training_score
    # print "part 5 different_six_performer_results is ", different_six_performer_results
    #
    # different_performers_file = open("part_5_different_performer_accuracies.txt", "w")
    # for key in different_six_performer_results:
    #     different_performers_file.write(str(key) + " Accuracy is: " + str(different_six_performer_results[key]) + "\n")
    # different_performers_file.close()
    #
    # plot_performance_gender(actors, actresses, act, act_different, actors_different, actresses_different)

    ## Part 6

    part_6_finite_difference(f_multiclass, df_multiclass)
