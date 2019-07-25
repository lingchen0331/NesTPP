# -*- coding: utf-8 -*-
# Author: chen
# Created at: 6/12/19 10:09 AM

import data_helper as dh
#import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
#import matplotlib.pyplot as plt
#import time

####################################
###       Hyper Parameters       ###
####################################
_E = 10**(-10)
SCALE = 0.001
MAX_ITER = 1000
MAX_FUN = 1000
PATH = '/nested_hawkes/'
PATH_movie = '/nested_hawkes/data/Avengers/'


def get_associated_replies(main_dataframe, reply_dataframe):
    """Get associated replies for all main stream events"""
    sub_id = list(main_dataframe.sub_id)
    
    reply_dataframe_list = []
    
    for i, x in enumerate(sub_id):
        pointers = []
        for j in range(len(reply_dataframe)):
            if reply_dataframe.iloc[j].link_id == x:
                pointers.append(j)
        
        reply_dataframe_list.append(reply_dataframe.iloc[pointers])
        print("{}, Done".format(i))
    
    return reply_dataframe_list


def get_number_of_previous_events(t, main_time_series, reply_time_series):
    """Generate the mark information for each time stamp for the main stream events"""
    main_time_nplist = np.array(main_time_series)
    index_list = main_time_nplist[np.where(main_time_nplist < t)]
    
    ii = [np.where(main_time_nplist == i)[0][0] for i in index_list]
    #print(ii)
    mark_info = np.array([len(np.where(np.array(reply_time_series[i]) < t)[0]) for i in ii])
    
    return mark_info 


def main_cif(t, main_time_series, reply_time_series, params):
    " Conditional intensity function of the main stream events with power law kernel.              "
    "                                                                                              "
    " The conditional intensity function is:                                                       "
    " lambda_main*(t) = mu_ma + sum(lambda_rep * p_i**gamma * (t - t_i + c)**(-(eta + 1)))         "
    "                                                                                              "
    " Parameters:                                                                                  "
    " - mu_ma corresponds to the baseline intensity of the main stream posts.                      "
    " - gamma represents the wrapping effect of the mark information                               "
    " - c denotes the regularization term to keep exponent bounded.                                "
    " - eta denotes the power law exponent for each previous event's influence                     "
    
    # Getting all the parameters
    # Base intensity
    mu_ma, mu_rep = params[0], params[1]
    # Parameters from reply post stream
    alpha, beta, delta = params[2], params[3], params[4]
    # Parameters from main post stream
    c, gamma, eta = params[5], params[6], params[7]
    
    mm = np.array(main_time_series)
    base = mu_ma
    
    mark_info = (get_number_of_previous_events(t, main_time_series, reply_time_series)+1) ** gamma
    reply_intensity = reply_cif(t, main_time_series, reply_time_series, mu_rep, alpha, beta, delta)
    
    #expo = (np.subtract(t + c, mm[np.where(mm < t)])*SCALE)**(-eta - 1)
    expo = (np.subtract(t + c, mm[np.where(mm < t)]))**(-eta - 1)
    return base + sum(mark_info * reply_intensity * expo)
    
    
def reply_cif(t, main_time_series, reply_time_series, mu_rep, alpha, beta, delta):
    " Conditional intensity function of a reply stream events with exponential kernel.             "
    "                                                                                              "
    " The conditional intensity function is:                                                       "
    " lambda*(t) = mu + sum(alpha*exp(-beta*(t - times[i])))                                       "
    "                                                                                              "
    " Parameters:                                                                                  "
    " - mu_rep corresponds to the baseline intensity of the reply stream posts.                    "
    " - alpha corresponds to the jump intensity, representing the jump in intensity upon arrival.  "
    " - beta is the decay parameter, governing the exponential decay of intensity.                 "
    " - delta represents the decaying influence brought by the associated main post.               "
    
    # index = np.where(np.array(main_time_series) == t)[0][0]
    main_time_nplist = np.array(main_time_series)
    #print(type(main_time_nplist), type(t))
    ii = main_time_nplist[np.where(main_time_nplist < t)]
    
    index_list = [np.where(main_time_nplist == i)[0][0] for i in ii]

    # Initialize a new index list
    reply_lists = []
    
    for i in index_list:
        reply_lists.append(
                np.array(reply_time_series[i])[np.where(np.array(reply_time_series[i]) < t)])
        
    cif_list = []
    
    for j in reply_lists:
        if j.size > 0:
            #influence_from_main = np.exp(-delta*(np.subtract(t, j[-1])*SCALE))
            #aa = sum(alpha*np.exp(-beta*(np.subtract(j[-1], j[np.where(j<j[-1])]))*SCALE))
            influence_from_main = np.exp(-delta*(np.subtract(t, j[-1])))
            aa = sum(alpha*np.exp(-beta*(np.subtract(j[-1], j[np.where(j<j[-1])]))))
            #print(influence_from_main, aa)
            cif_list.append(mu_rep + influence_from_main*aa)
        else:
            cif_list.append(mu_rep)
    
    return cif_list    
    

def rep_cif_simu(t, main_time, reply_stream, mu_rep, alpha, beta, delta):
    j = np.array(reply_stream)
    influence_from_main = np.exp(-delta*(np.subtract(t, main_time)))
    
    return mu_rep + influence_from_main*sum(alpha*np.exp(-beta*(np.subtract(t, j[np.where(j<t)]))))


def logLikelihood(params, main_time_series, reply_time_series, verbose=False):
    """Log-likelihood object function"""
    " The log-likelihood fucntion is composed with main post stream and reply post stream events   "
    "                                                                                              "
    " The ll function is:                                                                          "
    " sum(sum(log(lambda_main(t))+log(lambda_reply(t))))-Lambda_main(t_m)-sum(Lambda_reply(t_ni))  "
    
    mm = np.array(main_time_series)

    # Getting all the parameters
    # Base intensity
    mu_ma, mu_rep = params[0], params[1]
    # Parameters from reply post stream
    alpha, beta, delta = params[2], params[3], params[4]
    # Parameters from main post stream
    c, gamma, eta = params[5], params[6], params[7]
    
    """ Calculating the log-intensity function """
    log_cif_index = []
    for i in range(1, len(main_time_series)):
        log_cif_index.append(np.log(main_cif(
                main_time_series[i], main_time_series, reply_time_series, params)))
        log_cif_index.append(sum(np.log(
                reply_cif(main_time_series[i], main_time_series, reply_time_series, mu_rep, alpha, beta, delta))))

    #print(log_cif_index)
    log_cif = sum(log_cif_index)


    """ Calculating the compensator for main stream events """
    main_compensator = []
    #main_base = mu_ma * np.subtract(mm[-1], mm[0])*SCALE
    main_base = mu_ma * np.subtract(mm[-1], mm[0])
    for i, x in enumerate(main_time_series[:-1]):
        reply_intensity = reply_cif(x, main_time_series, reply_time_series, mu_rep, alpha, beta, delta)
        marks = (get_number_of_previous_events(x, mm, reply_time_series)+1) ** gamma
        #exp_term = np.subtract((eta*(c**eta))**(-1), eta**(-1)*(np.subtract(mm[-1]+c,x)*SCALE)**(-eta))
        exp_term = np.subtract((eta*(c**eta))**(-1), eta**(-1)*(np.subtract(mm[-1]+c,x))**(-eta))
        main_compensator.append(main_base + np.sum(reply_intensity*marks*exp_term))    
    
    main_compensator = sum(main_compensator)
    
    """ Calculating the compensator for reply stream events """
    base_rep = []
    for i in reply_time_series[:-1]:
        if np.array(i).size > 0:
            #base_rep.append(mu_rep*np.subtract(i[-1], i[0])*SCALE)
            base_rep.append(mu_rep*np.subtract(i[-1], i[0]))
        else:
            base_rep.append(0)
    
    exponent_rep = []
    for i, x in enumerate(reply_time_series[:-1]):
        kk = np.array(x)
        if kk.size > 0:
            #influence_from_main = np.exp(-delta*(np.subtract(kk[-1], mm[i])*SCALE))
            #exponent_rep.append(((alpha*influence_from_main)/beta)*sum(np.exp(-beta*(np.subtract(kk[-1],kk[np.where(kk<kk[-1])]))*SCALE)-1))
            influence_from_main = np.exp(-delta*(np.subtract(kk[-1], mm[i])))
            exponent_rep.append(((alpha*influence_from_main)/beta)*sum(np.exp(-beta*(np.subtract(kk[-1],kk[np.where(kk<kk[-1])])))-1))
        else:
            exponent_rep.append(0)
    
    compensator_reply = sum(base_rep) - sum(exponent_rep)
    
    #return main_compensator
    
    return -(log_cif - compensator_reply  - main_compensator)


def mle(main_time_series, reply_time_series, verbose=True):
    " Maximum-Likelihood Estimation for NTPP parameters"
    " Given a main stream sequence of observations and associated reply streams"
    
    # generate random parameter estimates for eight parameters
    params = np.random.uniform(0,1,size=8)
    print(params)
    # minimize the negative log-likelihood function
    res = minimize(logLikelihood, params, args=(main_time_series, reply_time_series, verbose), 
                   bounds=[(1e-3, None), (1.5e-3, 1e-2), (1e-5, None), (1e-5, None), (1e-5, None), 
                           (1e-5, None), (1e-5, None), (1e-5, None)],
                   method="L-BFGS-B",
                   options={"ftol": 1e-6, 
                            "maxcor": 50, 
                            "maxiter": MAX_ITER, 
                            "maxfun": MAX_FUN,
                            "disp":True})


    return (res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.x[6], res.x[7])


def thinning_simulation(T, params, main_history, reply_history):
    " Thinning algorithm to simulate nested Hawkes processes              "
    " Input:                                                              "
    " - T is the number of events that expects to be simulated            "
    " - params denotes the trained parameters from the given sequence     "
    " - main_history represents the main stream events history            "
    " - reply_history represents the reply stream events history          "
    "                                                                     "
    " Output:                                                             "
    " - The simulated main stream events time and the reply stream events "
    # Getting all the parameters
    # Base intensity
    mu_rep = params[1]
    # Parameters from reply post stream
    alpha, beta, delta = params[2], params[3], params[4]
    
    M = main_history.copy(); R = reply_history.copy()
    #print(M, R)
    index_a = len(main_history)
    
    t = M[-1] if len(M) > 0 else 0
    
    counter = 0
    while counter < T:
        
        # Generate reply stream events
        if counter > 0:
            for i in range(index_a, len(R)):
                R[i] += adaptive_thining(M[i], M[i], R[i], 300, mu_rep, alpha, beta, delta)
                #print(len(R[i]))
        
        # find new upper bound M
        upper_bound = main_cif(t, M, R, params)
        # generate next candidate point 
        next_candidate_point  = -(1/upper_bound)*np.log(np.random.uniform(0, 1))
        t += next_candidate_point

        # accept it with some probability: U[0, M]
        U = np.random.uniform(0, upper_bound)
        
        if (U <= main_cif(t, M, R, params)):
            M.append(t)
            R.append([])
            #print("The {} iteration simulation, Done".format(counter))
            counter += 1
        else:
            #print("The {} iteration simulation, Fail".format(counter))
            counter += 1
        print("Done")
    return M[index_a:], R[index_a:]


def adaptive_thining(t, main_time, reply_stream, time_range, mu_rep, alpha, beta, delta):
    "Thinning algorithm to simulate nested Hawkes processes "
    P = []; time_range = t+time_range; counter = 0
    
    while t < time_range and counter < 2:
        M = rep_cif_simu(t+_E, main_time, reply_stream, mu_rep, alpha, beta, delta)
        E = -(1/M)*np.log(np.random.uniform(0, 1))
        t += E
        #print(t)
        # accept it with some probability: U[0, M]
        U = np.random.uniform(0, M)

        if (U <= rep_cif_simu(t+_E, main_time, reply_stream, mu_rep, alpha, beta, delta)):
            P.append(t)
            counter += 1
    return P
    

def generating_training_time_series(lower_range, upper_range):
    # Read stream data
    _main_stream = dh.load_json(PATH+'james.json') # All main stream data.
    #reply_stream = dh.load_json('james_reply.json') # All reply stream data.
    _reply_stream = [dh.load_json(PATH+"james/{}.json".format(i)) for i in range(len(_main_stream))]
    
    # main posts time series
    main_stream_time_series = dh.main_stream_time_lists(_main_stream)[lower_range:upper_range]
    reply_stream_time_series = dh.reply_stream_time_lists(_reply_stream)[lower_range:upper_range]

    return main_stream_time_series, reply_stream_time_series


def generating_reply_time_series(upper_range, number):
    # Read stream data
    _main_stream = dh.load_json(PATH+'james.json') # All main stream data.
    #reply_stream = dh.load_json('james_reply.json') # All reply stream data.
    _reply_stream = [dh.load_json(PATH+"james/{}.json".format(i)) for i in range(len(_main_stream))]
    
    # main posts time series
    main_stream_time_series = dh.main_stream_time_lists(_main_stream)[upper_range:upper_range+number]
    reply_stream_time_series = dh.reply_stream_time_lists(_reply_stream)[upper_range:upper_range+number]
    
    for i in range(len(reply_stream_time_series)):
        temp = np.array(reply_stream_time_series[i])
        reply_stream_time_series[i] = temp[np.where(temp < main_stream_time_series[-1])]
        
    return main_stream_time_series, reply_stream_time_series


def cross_validation(testing_size):
    prediction_total_main = []; prediction_total_reply = []
    cross_vali_list = []
    for i in range(0, 6597, 200):
        cross_vali_list.append(i)
    
    for j in range(len(cross_vali_list)-1):
        main_stream_time_series, reply_stream_time_series = generating_training_time_series(cross_vali_list[j], cross_vali_list[j+1])
        testing_main_stream, testing_reply_stream = generating_reply_time_series(cross_vali_list[j+1], testing_size)
        
        params = mle(main_stream_time_series, reply_stream_time_series)
        print("Get parameters...")
        
        prediction_main = []; prediction_reply = []
        for i in range(100):
            pred_main, pred_reply = thinning_simulation(testing_size, params, main_stream_time_series, reply_stream_time_series)
            #mae = mean_absolute_error(testing_main_stream, pred_main)
            prediction_main.append(pred_main)
            #print("The MAE at {}'s iteration is: {}".format(i, mae))
            #MAE_main.append(mae)
            prediction_reply.append(pred_reply)
        #MAE_total.append(MAE_main)
        prediction_total_main.append(prediction_main)
        prediction_total_reply.append(prediction_reply)
        
    return prediction_total_main, prediction_total_reply



#prediction_total, prediction_total_reply = cross_validation(20)

main_stream_time_series, reply_stream_time_series = generating_training_time_series(0, 1000)
testing_main_stream, testing_reply_stream = generating_reply_time_series(1000, 20)

params = mle(main_stream_time_series, reply_stream_time_series)
print("\n###############################\n")
print(params)

'''
print("Start simulation...")
#params = (0.001, 0.01, 0.21061002151202762, 0.0021289880252872502, 0.0010901802421204004, 3.5399332798994396, 1e-05, 3.255250170787572)

MAE_main = []
prediction_main, prediction_reply = [], []
for i in range(10):
    #pred_main, pred_reply = thinning_simulation(10, _params, main_stream_time_series, reply_stream_time_series)
    pred_main, pred_reply = thinning_simulation(20, params, main_stream_time_series, reply_stream_time_series)
    mae = mean_absolute_error(testing_main_stream, pred_main)
    prediction_main.append(pred_main)
    prediction_reply.append(pred_reply)
    print("The MAE at {}'s iteration is: {}".format(i, mae))
    MAE_main.append(mae)

print("The mean MAE is :", np.mean(MAE_main))



cross_vali_list = []
for i in range(4596, 6597, 200):
    cross_vali_list.append(i)

mae_prediction = []
for j in range(len(prediction_total)):
    bb= []
    aa = np.array(prediction_total[j][0])
    testing_main_stream, testing_reply_stream = generating_reply_time_series(cross_vali_list[j+1], 20)
    for i in range(1, 100):
        aa += np.array(prediction_total[j][i])
    aa = (aa/100).tolist()
    for k in range(len(aa)):
        bb.append(np.abs(aa[k]-testing_main_stream[k]))
    mae_prediction.append(bb)

mae = []
for j in range(20):
    temp = []
    for i in range(len(mae_prediction)):
            temp.append(mae_prediction[i][j])
    mae.append(temp)

mae_modify = np.array([np.mean(i) for i in mae])/3600


for i in range(len(mae_modify)):
    with open("Output.txt", "w") as text_file:
        print(f"The Mean Absolute Error is {mae}: \n", file=text_file)



mae_prediction_reply = []
for j in range(len(prediction_total_reply)):
    aa = np.array([len(i) for i in prediction_total_reply[j][0]])
    testing_main_stream, testing_reply_stream = generating_reply_time_series(1500, 20)
    testing_reply_stream = np.array([len(i) for i in testing_reply_stream])
    for i in range(1, 100):
        aa += np.array([len(i) for i in prediction_total_reply[j][i]])
    aa = (aa/100)
    
    mae = np.abs(aa - testing_reply_stream)
    print(mae)
    mae_prediction_reply.append(mae)
    
tttt = []
for j in range(20):
    temp = []
    for i in range(len(mae_prediction_reply)):
            temp.append(mae_prediction_reply[i][j])
    tttt.append(temp)
    
mae_modify_reply = np.array([np.mean(i) for i in tttt])


print(mae_modify)

'''
