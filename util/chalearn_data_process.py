import pandas as pd
import pathlib
from face_preprocessing.align import FaceAligner

if __name__ == '__main__':
    BASE_PATH = pathlib.Path("data")
    train_or_valid = "valid"
    input_file = f"{train_or_valid}_gt.csv"
    output_file = f"{train_or_valid}_gt_int.csv"
    output_labels_path = pathlib.Path(BASE_PATH, train_or_valid)
    input_file_path = pathlib.Path(BASE_PATH, f"{train_or_valid}_gt/{input_file}")
    output_file_path = pathlib.Path(BASE_PATH, f"{train_or_valid}_gt/{output_file}")
    data_df = pd.read_csv(input_file_path)
    if output_file_path.is_file():
        print(f"already converted {input_file} to int")
    else:
        print(f"Converting {input_file} prediction column to int from float")
        data_df = data_df.astype({'mean': 'int32'})
        data_df.to_csv(f"data/{train_or_valid}_gt/{train_or_valid}_gt_int.csv", index=False)
    data_df = pd.read_csv(output_file_path)
    data_df.pop("stdv")
    print(data_df)

    age_classes = set(data_df['mean'].tolist())
    for age in age_classes:
        try:
            pathlib.Path(output_labels_path, f"{age}").mkdir()
        except FileExistsError:
            print(f"{output_labels_path}\{age} already exists")
    data_df = data_df.set_index('image').T.to_dict('list')
    for img, v in data_df.items():
        age = str(v[0])
        dest = pathlib.Path(output_labels_path, age, img)
        source = pathlib.Path(output_labels_path, img)
        if not dest.exists():
            source.replace(dest)
