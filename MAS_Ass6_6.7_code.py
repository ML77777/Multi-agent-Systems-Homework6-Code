
import numpy as np
import scipy.stats as sp

n = 1000
samples = np.random.uniform(-5,5,n)
sum = 0


def sum_question_one(samples):
    sum = 0
    for sample in samples.tolist():
        normal_density = sp.norm(0,1).pdf(sample)
        uniform_density = sp.uniform(-5,10).pdf(sample)
        #print("Sample: {}, normal desntiy: {}, uniform density {}".format(sample, normal_density, uniform_density))
        sum += sample**2 * (normal_density/(uniform_density))
    return sum

def question_one():
    sum = sum_question_one(samples)
    mean = sum/n
    print("Sum: {}, Mean: {}".format(sum, mean))

def print_true_mean():
    samples = np.random.normal(0,1,1000)
    sum = 0
    for sample in samples.tolist():
        sum += sample**2
    mean = sum / 1000
    print("True sum: {}, True Mean: {}".format(sum, mean))


question_one()
print_true_mean()
