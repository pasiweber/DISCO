import pandas as pd


def check_dir_exists(dir_name):
    import os
    return os.path.isdir(dir_name)


def create_dir(dir_name):
    if not check_dir_exists(dir_name):
        import os
        os.mkdir(dir_name)


def save_data(results, exp_name):
    print('Saving data...')
    dataframe = pd.DataFrame(results)
    import datetime
    timestamp = datetime.datetime.now().timestamp()
    formatted_string = datetime.datetime.fromtimestamp(timestamp).strftime("%m-%d-%H-%M-%S")
    dir_name = "results/{}".format(exp_name)
    create_dir(dir_name)
    dataframe.to_csv("{}/combined_{}.csv".format(dir_name, formatted_string))
    print('saving finished')


def get_latest(exp_name):
    import os
    import glob
    list_of_files = glob.glob('results/{}/*'.format(exp_name))  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file