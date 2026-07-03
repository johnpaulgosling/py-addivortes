# This function uses code from the articles below to give a benchmark test for run times on
# 4 different examples using the AddiVortes algorithm. Each example fits the model, and then
# finds the mean predictions for each data point.

def tests(nDigits = 3, nTimes = 1):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    from addivortes import AddiVortesRegressor
    import pandas as pd
    import time
    import numpy as np
    from threadpoolctl import threadpool_info
    print(threadpool_info())

    def test1():
        x_train = pd.read_csv("./benchmarks/datasets/boston/x_train.csv")
        y_train = pd.read_csv("./benchmarks/datasets/boston/y_train.csv").iloc[:, 0].to_numpy()
        x_test = pd.read_csv("./benchmarks/datasets/boston/x_test.csv")
        y_test = pd.read_csv("./benchmarks/datasets/boston/y_test.csv").iloc[:, 0].to_numpy()

        start_time_fit = time.perf_counter()
        model = AddiVortesRegressor(
            n_tessellations = 200,
            total_mcmc_iter=2000,
            burn_in = 200
        )
        model.fit(x_train, y_train)
        end_time_fit = time.perf_counter()

        start_time_preds = time.perf_counter()
        preds = model.predict(x_test)
        end_time_preds = time.perf_counter()

        d = {"Fit time": [round(end_time_fit-start_time_fit, nDigits)],
              "Prediction time": [round(end_time_preds-start_time_preds, nDigits)],
              "In-sample RMSE": [round(model.in_sample_rmse_, nDigits)],
              "Out-of-sample RMSE": [float(round(np.sqrt(np.mean((y_test - preds) ** 2)), nDigits))]}
        return(pd.DataFrame(data=d))

    def test2():
        x_train = pd.read_csv("./benchmarks/datasets/synthetic/x_train.csv")
        y_train = pd.read_csv("./benchmarks/datasets/synthetic/y_train.csv").iloc[:, 0].to_numpy()
        x_test = pd.read_csv("./benchmarks/datasets/synthetic/x_test.csv")
        y_test = pd.read_csv("./benchmarks/datasets/synthetic/y_test.csv").iloc[:, 0].to_numpy()

        start_time_fit = time.perf_counter()
        model = AddiVortesRegressor(n_tessellations = 50)
        model.fit(x_train, y_train)
        end_time_fit = time.perf_counter()

        start_time_preds = time.perf_counter()
        preds = model.predict(x_test)
        end_time_preds = time.perf_counter()

        d = {"Fit time": [round(end_time_fit-start_time_fit, nDigits)],
              "Prediction time": [round(end_time_preds-start_time_preds, nDigits)],
              "In-sample RMSE": [round(model.in_sample_rmse_, nDigits)],
              "Out-of-sample RMSE": [float(round(np.sqrt(np.mean((y_test - preds) ** 2)), nDigits))]}
        return(pd.DataFrame(data=d))

    def test3():
        x_train = pd.read_csv("./benchmarks/datasets/spherical/x_train.csv")
        y_train = pd.read_csv("./benchmarks/datasets/spherical/y_train.csv").iloc[:, 0].to_numpy()
        x_test = pd.read_csv("./benchmarks/datasets/spherical/x_test.csv")
        y_test = pd.read_csv("./benchmarks/datasets/spherical/y_test.csv").iloc[:, 0].to_numpy()

        start_time_fit = time.perf_counter()
        model = AddiVortesRegressor(
            n_tessellations = 50,
            total_mcmc_iter=500, # Using default R values for mcmc iter and burn in
            burn_in = 100,
            metric="spherical"
        )
        model.fit(x_train, y_train)
        end_time_fit = time.perf_counter()

        start_time_preds = time.perf_counter()
        preds = model.predict(x_test)
        end_time_preds = time.perf_counter()

        d = {"Fit time": [round(end_time_fit-start_time_fit, nDigits)],
              "Prediction time": [round(end_time_preds-start_time_preds, nDigits)],
              "In-sample RMSE": [round(model.in_sample_rmse_, nDigits)],
              "Out-of-sample RMSE": [float(round(np.sqrt(np.mean((y_test - preds) ** 2)), nDigits))]}
        return(pd.DataFrame(data=d))

    def test4():
        x_train = pd.read_csv("./benchmarks/datasets/categorical/x_train.csv")
        y_train = pd.read_csv("./benchmarks/datasets/categorical/y_train.csv").iloc[:, 0].to_numpy()
        x_test = pd.read_csv("./benchmarks/datasets/categorical/x_test.csv")
        y_test = pd.read_csv("./benchmarks/datasets/categorical/y_test.csv").iloc[:, 0].to_numpy()

        start_time_fit = time.perf_counter()
        model = AddiVortesRegressor(
            n_tessellations = 50,
            total_mcmc_iter=500, # Using default R values for mcmc iter and burn in
            burn_in = 100,
            cat_scaling=1.0
            )
        model.fit(x_train, y_train)
        end_time_fit = time.perf_counter()

        start_time_preds = time.perf_counter()
        preds = model.predict(x_test)
        end_time_preds = time.perf_counter()

        d = {"Fit time": [round(end_time_fit-start_time_fit, nDigits)],
              "Prediction time": [round(end_time_preds-start_time_preds, nDigits)],
              "In-sample RMSE": [round(model.in_sample_rmse_, nDigits)],
              "Out-of-sample RMSE": [float(round(np.sqrt(np.mean((y_test - preds) ** 2)), nDigits))]}
        return(pd.DataFrame(data=d))
    
    def to_csv_string(*args):
        return (",".join(str(x) for x in args) + "\n")
    
    dfs = []
    times_dfs = []
    errors_dfs = []

    for i in range(0, nTimes):
        results = pd.concat([test1(), test2(), test3(), test4()], axis=0)
        print("Iteration " + str(i) + " completed")
        results.round(nDigits)
        with open("benchmarks/py-testlog.csv", "a") as f:
            f.write(to_csv_string(*results.to_numpy().flatten()))
        dfs.append(results)
        
    for results in dfs:
        results.index = ["Test 1", "Test 2", "Test 3", "Test 4"]
        times = results.iloc[:, :2]
        errors = results.iloc[:, 2:]
        times["Sum"] = times.sum(axis=1)
        times.loc["Sum"] = times.sum(axis=0)
        errors["Sum"] = errors.sum(axis=1)
        errors.loc["Sum"] = errors.sum(axis=0)
        times_dfs.append(times)
        errors_dfs.append(errors)
    avg_times = sum(times_dfs) / len(times_dfs)
    avg_errors = sum(errors_dfs) / len(errors_dfs)
    return [avg_times.round(nDigits), avg_errors.round(nDigits)]

results = tests(3, 1)
print(results[0])
print(results[1])