# NTPP: Nested Temporal Point Process

A nested point process in modeling social activities on online social forums, 
which is a a class that models the arrival in time of random events and their 
interaction with the state of a system.  


This repository contains three components:
- A comprehensive dataset of topic stream data and associated reply data from Reddit (retrieved by [Pushshift API](https://github.com/pushshift/api)).
- The full code of fitting and training all the parameters described in NTPP.py.
- The adaptive simulation step of utilizing fitted parameters.


## Dependencies
This code is written in Python. To use it you will need:
- Numpy - 1.16.2
- Scipy - 1.2.1
- pandas - 0.23.4

It is recommended to use [Anaconda](https://www.anaconda.com/) since it includes all the Python-related dependencies

## Usage
### Data
The data used in this project can be downloaded from this
 [link](https://drive.google.com/open?id=1snS-6f_0EoxcMfN9_Hiw9NxkTX8ha-m5).
 
We also present the data preprocessing program *data_cleaner.py* which provides the necessary procedure of producing the model-required Reddit data retrived from [Pushshift API](https://github.com/pushshift/api).

### Train Models
To train the model, try to run the ntpp.py.
```
python ntpp.py
```

Note that the parameter value ranges are hyper-parameters, and different range 
may result different performance in different dataset, be sure to tune 
hyper-parameters carefully. 

 
