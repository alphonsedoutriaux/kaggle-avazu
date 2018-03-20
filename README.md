

# Avazu Click-Through-Rate - Kaggle challenge
#### Author : Alphonse Doutriaux - March 2018

Original train and tests sets can be found [here](https://www.kaggle.com/c/avazu-ctr-prediction).

For this Kaggle challenge, I used an AWS t2-2xlarge instance, which is not enough to deal with the 40M+ lines of the data set.

#### 1. Random import of 1 to 2M lines, depending on the model used 

#### 2. Preprocessing
2.1. Features creation : *weekday*, *hour*, *surface* (C15 x C16)

2.2. Features dropping : *site_id*, *site_domain*, *app_id*, *app_domain*, *device_id*, *device_ip*, *device_model*

2.3. in C20 column, -1 values have been replaced with the median of the column

2.4. One-hot-encoding (using get_dummies) of *device_type*, *device_conn_type*, *site_category*, *app_category*, *banner_pos*, *C18*  


#### 3. Models : various models have been tested but XGBoost showed the best results in this challenge. Hyperparameters were set using sklearn's GridsearchCV function.

#### 4. Results :
best log_loss score on kaggle.com (private score) : 0.4104



