import pandas as pd

def save_data(results, exp_name):
    print('Saving data...')
    dataframe = pd.DataFrame(results)
    import datetime
    timestamp = datetime.datetime.now().timestamp()
    formatted_string = datetime.datetime.fromtimestamp(timestamp).strftime("%m-%d-%H-%M-%S")
    dataframe.to_csv("results/{}/combined_{}.csv".format(exp_name,formatted_string))
    print('saving finished')