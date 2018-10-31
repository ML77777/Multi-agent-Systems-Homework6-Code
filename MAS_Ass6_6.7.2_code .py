
import numpy as np
import scipy.stats as sp
import math
import scipy.integrate as integrate

n = 1000
samples = np.random.uniform(-5,5,n)
sum = 0

def calculate_p_density(sample):
    if sample > 1 or sample < -1:
        return 0.0
    numerator = 1 + math.cos(np.pi*sample)
    density = numerator / 2
    return density

def sum_question_two(samples):
    sum = 0
    for sample in samples.tolist():
        p_density = calculate_p_density(sample)
        uniform_density = sp.uniform(-5,10).pdf(sample)
        #print("Sample: {}, normal desntiy: {}, uniform density {}".format(sample, normal_density, uniform_density))
        sum += sample**2 * (p_density/uniform_density)
    return sum

def real_expectation():
    n = 1000
    probabilities = [0]
    range_intervals = np.linspace(-1,1,100)
    #print(range_intervals)
    for i in range(99):
        lower_limit = range_intervals[i]
        upper_limit = range_intervals[i+1]
        result = integrate.quad(lambda x: (1 + math.cos(np.pi*x))/2, lower_limit,upper_limit)[0]
        probabilities.append(result)
    #print(probabilities)
    #print(math.fsum(probabilities))

    samples = np.random.choice(range_intervals, 1000, replace=True, p=probabilities)
    #print(samples)
    sum = 0
    for sample in samples:
        sum += sample**2
    mean = sum/n
    print("True sum: {}, True mean: {}".format(sum,mean))
    return mean

def question_two():
    sum = sum_question_two(samples)
    mean = sum/n
    print("Sum: {}, Mean: {}".format(sum, mean))
    return mean

importance_sampling_e_value = question_two()
direct_sampling_e_value = real_expectation() #Dont know if this is correct either
print("Difference: {}".format(direct_sampling_e_value - importance_sampling_e_value))



