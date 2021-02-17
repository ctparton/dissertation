from scipy.io import loadmat
from pathlib import Path
import pandas as pd
import csv
if __name__ == '__main__':
    BASE_PATH = Path('C:\\Users\\CallumDesk\\Downloads\\wiki_crop')
    annots = loadmat(Path(BASE_PATH / 'wiki.mat'))
    data = [[row.flat[0] for row in line] for line in annots['wiki'][0]]
    # df_train = pd.DataFrame(data, columns=['dob', 'photo_taken', 'full_path', 'gender', 'name', 'face_location', 'face_score', 'second_face_score'])
    data_dict = loadmat(Path(BASE_PATH / 'wiki.mat'))
data_array = data_dict['wiki']
data_array = data_array.transpose(1, 0)
df = pd.DataFrame(data_array)
print(df)

