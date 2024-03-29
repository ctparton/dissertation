from scipy.io import loadmat
from pathlib import Path
import pandas as pd
import csv


def remove_poor_samples(no_of_files, dict_from_mat, base_path):
    """
    Removes poor samples for IMDB-Wiki

    :param no_of_files: number of samples in the dataset
    :param dict_from_mat: the dictionary created from the MatLab file
    :param base_path: where to save the correct samples
    :return:
    """
    correct_samples = 0
    for i in range(no_of_files):
        # birth year
        birth_year = int(dict_from_mat[0][0][i] / 365)
        # photo taken
        photo_taken = dict_from_mat[1][0][i]
        path = dict_from_mat[2][0][i][0]
        # Face score
        face_score = dict_from_mat[6][0][i]
        # Sec face score
        sec_face_score = dict_from_mat[7][0][i]

        age = photo_taken - birth_year

        face_score = str(face_score)
        sec_face_score = str(sec_face_score)

        # Check for Inf; if true, implies that there isn't a face in the image
        if 'n' not in face_score:
            # Check for NaN; if True, no second face in image
            if 'a' in sec_face_score:
                # Check age
                if 0 <= age <= 101:
                    correct_samples += 1

                    file = f"{Path(path).stem}_{age}{Path(path).suffix}"
                    try:
                        source = Path(base_path / path)
                        dest = Path(base_path / 'all' / file)
                        if not dest.exists():
                            source.replace(dest)
                        else:
                            print(f"dest {dest} exists")
                    except FileExistsError:
                        print(f"'All' directory already exists in {base_path}")
        else:
            print(f"Bad data {Path(base_path / path)}")

    print(correct_samples)


def count_total_files(dir_path):
    """
    Counts the total files in a directory
    :param dir_path: directory to the images
    :return: number of files
    """
    total_files = 0
    has_subdir = not any([x.is_file() for x in dir_path.iterdir() if x.suffix != '.mat'])
    print(f"Subdirectories in {dir_path} = {has_subdir}")
    for age_folder in dir_path.iterdir():
        if not has_subdir:
            total_files = total_files + 1
        else:
            if age_folder.is_dir():
                for _ in age_folder.iterdir():
                    total_files = total_files + 1

    return total_files


def build_csv(dir_path, train_size):
    """
    Creates a CSV, which can be read in during training

    :param dir_path: path to save the csv
    :param train_size: portion of samples to label as training data
    :return: void
    """
    csv_label_file = Path(dir_path / 'imdb_wiki_labels_t60.csv')
    if csv_label_file.is_file() and csv_label_file.exists():
        print(f"{csv_label_file} already exists")
    else:
        with open(csv_label_file, mode='w', newline="") as csvfile:
            fieldnames = ['image', 'age', 'partition']
            label_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            label_writer.writeheader()
            count = 0
            partition = "train"
            for item in dir_path.iterdir():
                if count > train_size:
                    partition = "valid"
                parts = item.stem.split("_")
                print(f"Writing Item: {item.name} Age: {parts[len(parts) - 1]} Partition {partition}")
                label_writer.writerow({'image': item.name, 'age': parts[len(parts) - 1], 'partition': partition})
                count = count + 1
        # print(count)

    with open(csv_label_file, mode='r') as csvfile:
        output_file = csv.reader(csvfile)
        row_count = sum(1 for row in output_file)
        print(f"Total Rows {row_count}")

def load_label_data():
    """
    Loads matlab files for the IMDB and wiki datasets into a dict

    :return: Tuple of dictionaries
    """
   return (loadmat(BASE_PATH_IMDB / 'imdb.mat')['imdb'][0][0], loadmat(BASE_PATH_WIKI / 'wiki.mat')['wiki'][0][0])

if __name__ == '__main__':
    BASE_PATH_IMDB = Path('C:\\Users\\CallumDesk\\Downloads\\imdb_crop')
    BASE_PATH_WIKI = Path('C:\\Users\\CallumDesk\\Downloads\\wiki_crop')
    PROCESSED_FILE_PATH = (Path('C:\\Users\\CallumDesk\\Downloads\\imdb_wiki_processed'))
    train_dest = Path(BASE_PATH_IMDB / 'train')
    val_dest = Path(BASE_PATH_IMDB / 'validation')

    imdb_total_files = count_total_files(BASE_PATH_IMDB)
    wiki_total_files = count_total_files(BASE_PATH_WIKI)
    print(f"IMDB Total: {imdb_total_files} WIki Total: {wiki_total_files}")
    imdb_metadata, wiki_metadata = load_label_data()

    print(remove_poor_samples(imdb_total_files, imdb_metadata, BASE_PATH_IMDB))
    print(remove_poor_samples(wiki_total_files, wiki_metadata, BASE_PATH_WIKI))
    processed_total_files = count_total_files(PROCESSED_FILE_PATH)
    print(f"Total processed files: {processed_total_files}")
    print(f" 70% train split: {int(processed_total_files * 0.60)}")
    print(build_csv(PROCESSED_FILE_PATH, int(processed_total_files * 0.60)))
