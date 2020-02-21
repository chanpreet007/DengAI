from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

def find_best_params_histGradientBoosting(X_train, y_train, X_val, y_val, param1_list, param2_list, param3_list, param4_list):

    param1_name = "max_iter"
    param2_name = "max_leaf_nodes"
    param3_name= "max_depth"
    param4_name= "max_bins"
    print("mean_absolute_error [{0},{1}]: mean absolute error".format(param1_name, param2_name))

    scores_mean_absolute_error = np.zeros((len(param1_list), len(param2_list)))
    best_score_mean_absolute_error = float("inf")
    best_parameters = [0, 0]

    for i in range(len(param1_list)):
        for k in range(len(param2_list)):
            for t in range(len(param3_list)):
                for j in range(len(param4_list)):
                    
                    param1 = param1_list[i]
                    param2 = param2_list[k]
                    param3 = param3_list[t]
                    param4 = param4_list[j]
                    regressor = HistGradientBoostingRegressor(loss='least_squares',
                                     learning_rate=0.05,
                                     max_iter=param1,
                                     max_leaf_nodes=param2,
                                     max_depth=param3,
                                     min_samples_leaf=20,
                                     l2_regularization=0.6,
                                     max_bins=param4,
                                     scoring=None,
                                     validation_fraction=0.6,
                                     n_iter_no_change=None,
                                     tol=1e-07, 
                                     verbose=0,
                                     random_state=0)

                    regressor.fit(X_train, y_train)
                    predictions = regressor.predict(X_val)

                    mean_absolute_error_ = mean_absolute_error(y_val, predictions)
                    print("mean_absolute_error [{0},{1},{2},{3}]: {4}".format(param1, param2, param3,param4,                                         mean_absolute_error_))
                    scores_mean_absolute_error[i, j] = mean_absolute_error_

                    if mean_absolute_error_ < best_score_mean_absolute_error:
                    
                        best_score_mean_absolute_error = mean_absolute_error_
                        best_parameters = [param1, param2, param3, param4]
            
            

                

    print("\n-------------------------")
    print("best_mean_absolute_error: {0}".format(best_score_mean_absolute_error))
    print("{0}: {1}".format(param1_name, best_parameters[0]))
    print("{0}: {1}".format(param2_name, best_parameters[1]))
    print("{0}: {1}".format(param3_name, best_parameters[2]))
    print("{0}: {1}".format(param4_name, best_parameters[3]))
    print("-------------------------\n")

    best_param1 = best_parameters[0]
    best_param2 = best_parameters[1]
    best_param3 = best_parameters[2]
    best_param4 = best_parameters[3]
    return best_param1, best_param2,best_param3,best_param4, scores_mean_absolute_error

def find_best_params_RandomForest(X_train, y_train, X_val, y_val, param1_list, param2_list):

    param1_name = "n_estimators"
    param2_name = "max_depth"

    print("mean_absolute_error [{0},{1}]: mean absolute error".format(param1_name, param2_name))

    scores_mean_absolute_error = np.zeros((len(param1_list), len(param2_list)))
    best_score_mean_absolute_error = float("inf")
    best_parameters = [0, 0]

    for i in range(len(param1_list)):
        for j in range(len(param2_list)):

            param1 = param1_list[i]
            param2 = param2_list[j]

            regressor = RandomForestRegressor(n_estimators=param1,
                                              max_depth=param2,
                                              criterion='mse',
                                              random_state=0,
                                              warm_start = True,
                                              n_jobs=8)

            regressor.fit(X_train, y_train)
            predictions = regressor.predict(X_val)

            mean_absolute_error_ = mean_absolute_error(y_val, predictions)
            print("mean_absolute_error [{0},{1}]: {2}".format(param1, param2, mean_absolute_error_))
            scores_mean_absolute_error[i, j] = mean_absolute_error_

            if mean_absolute_error_ < best_score_mean_absolute_error:
                best_score_mean_absolute_error = mean_absolute_error_
                best_parameters = [param1, param2]

    print("\n-------------------------")
    print("best_mean_absolute_error: {0}".format(best_score_mean_absolute_error))
    print("{0}: {1}".format(param1_name, best_parameters[0]))
    print("{0}: {1}".format(param2_name, best_parameters[1]))
    print("-------------------------\n")

    best_param1 = best_parameters[0]
    best_param2 = best_parameters[1]

    return best_param1, best_param2, scores_mean_absolute_error



