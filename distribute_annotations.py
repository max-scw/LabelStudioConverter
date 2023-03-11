import pathlib as pl
import shutil

from typing import Union, List

import timeit
import random
def get_files(path_to_directory: Union[str, pl.Path],
              extension: str = ".txt",
              recursive: bool = False) -> List[pl.Path]:
    # ensure pathlib object
    path_to_directory = pl.Path(path_to_directory)
    if not path_to_directory.is_dir():
        raise ValueError(f"Provided path '{path_to_directory.as_posix()}' is not a directory.")

    # pathlib extensions should start with a dot
    if extension[0] != ".":
        extension = "." + extension

    regex = "*"
    if recursive:
        regex = "**/*"
    return list(path_to_directory.glob(regex + extension))


def keep_files_in_list(list_of_files: List[List[Union[str, pl.Path]]]) -> List[List[pl.Path]]:
    # (loop over list of files and) extract file names
    list_of_files_stem = []
    for lof in list_of_files:
        list_of_files_stem.append([pl.Path(fl).stem for fl in lof])

    # sort
    list_of_files_to_keep = [[] for _ in range(len(list_of_files_stem))]
    # extract list of paths and file names
    lof = list_of_files[0]
    lof_stem = list_of_files_stem[0]
    for fl, fl_stem in zip(lof, lof_stem):
        if all([fl_stem in list_of_files_stem[j] for j in range(1, len(list_of_files_stem))]):
            for i in range(0, len(list_of_files_stem)):
                for j in range(len(list_of_files[i])):
                    if fl_stem == list_of_files_stem[i][j]:
                        list_of_files_to_keep[i].append(list_of_files[i][j])
                        break
    return list_of_files_to_keep


if __name__ == "__main__":
    path_to_annotation_files = pl.Path("export")
    path_to_image_files = pl.Path(r"C:\Users\schwmax\Proj\Coding\YOLOv7_scw\dataset\NEW")

    annotation_files_all = get_files(path_to_annotation_files, extension=".txt")

    for fldr in ["Trn", "Tst", "Val"]:
        image_files = get_files(path_to_image_files / fldr, extension=".bmp")

        # ensure that file names are present in both lists
        image_files, annotation_files = keep_files_in_list([image_files, annotation_files_all])
        # copy annotations files to image files
        for (fl_img, fl_ann) in zip(image_files, annotation_files):
            source = pl.Path(fl_ann).as_posix()
            destination = pl.Path(fl_img).with_name(pl.Path(fl_ann).name).as_posix()
            # copy file
            shutil.copy(source, destination)

        # save data
        with open(f"{fldr}.txt", "w") as fid:
            for ln in image_files:
                fid.write(pl.Path(ln).as_posix() + "\n")
        print(f"done {fldr}.txt")




