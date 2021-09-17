import pandas as pd 
import os

data_path = '../data/'
feature_path = os.path.join(data_path, 'feature/teacher/')
if not os.path.exists(feature_path):
    os.makedirs(feature_path)


log_file = os.path.join(os.environ.get('OUTPUT_DIR'),'example/OPDT_b/test/')

files = os.listdir(log_file)

for f in files:
    if f.endswith(".log"):
        df = pd.read_pickle(log_file + f)

        #df['datetime'] = df.index.get_level_values(1).map(lambda x: x[1])
        df['datetime'] = df.index.get_level_values(1)
        df.set_index('datetime', append=True, drop=True, inplace=True)
        action = df['action']
        action = action.reset_index(level=1, drop=True)
        action.index = action.index.map(lambda x: (x[0], x[1], x[2].time()))
        action = action.unstack().iloc[:, ::30] * 2
        action = action.fillna(0)
        train_action = action.astype("int")
        final = train_action
        final.to_pickle(feature_path + f[:-4] + '.pkl')