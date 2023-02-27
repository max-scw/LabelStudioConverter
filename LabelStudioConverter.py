import json
import pathlib as pl
import re
import datetime

from typing import Union, List, Tuple


class LabelStudioAnnotation:
    def __init__(self,
                 path_to_annotations: Union[str, pl.Path],
                 decimal_precision: int = 0
                 ) -> None:
        # ensure pathlib object
        path_to_annotations = pl.Path(path_to_annotations)
        if not path_to_annotations.exists():
            raise ValueError(f"Annotation file {path_to_annotations.as_posix()} does not exist!")
        # store path as local variable
        self.path = path_to_annotations
        # store precision value (if it is valid)
        self.decimal_precision = decimal_precision if decimal_precision > 0 else 0
        # load file
        with open(self.path, "r") as fid:
            self.data = json.load(fid)

    def _iterfiles(self) -> dict:
        # label-studio organizes annotations per file => loop over all files
        for fl in self.data:
            yield fl

    @staticmethod
    def _iterannotations(file) -> dict:
        # process the annotation itself
        annotations = file["annotations"][0]["result"]
        for an in annotations:
            yield an

    def _iterlabels(self, file) -> Tuple[str, str, dict]:
        for an in self._iterannotations(file):
            annotation_type = an["type"]
            value = an["value"]
            labels = value[annotation_type]
            for lbl in labels:  # FIXME: why are labels a list?
                yield lbl, annotation_type, value

    def _to_decimal_precision(self, num: Union[float, int], precision: int = None) -> Union[float, int]:
        if precision is None:
            precision = self.decimal_precision
        out = round(num, precision)
        if precision == 0:
            out = int(out)
        return out

    @staticmethod
    def _istypekeypoint(annotation_type: str) -> bool:
        return True if re.match("keypoint", annotation_type, re.IGNORECASE) else False

    @staticmethod
    def _istypebbox(annotation_type: str) -> bool:
        return True if re.match("rectangle", annotation_type, re.IGNORECASE) else False

    def _extract_original_size(self, file: dict) -> Tuple[int, int]:
        # process the annotation itself
        annotations = file["annotations"][0]["result"]
        # ensure uniqueness
        values = self.ensure_uniqueness_of_annotations(annotations, ["original_width", "original_height"])
        return values["original_width"], values["original_height"]

    def _sacle_to_abs_coordinates(self, val: List[float], original_width: int, original_height: int) -> List[int]:
        num = len(val)

        num = int(num / 2)
        return [self._to_decimal_precision(el / 100 * fct)
                for el, fct in zip(val, [original_width, original_height] * num)]

    def _iter_images(self, path_to_files: Union[str, pl.Path] = None) -> Tuple[pl.Path, Tuple[int, int], dict]:
        # label-studio organizes annotations per file => loop over all files
        for fl in self._iterfiles():
            # strip the preceding hash code
            filename = extract_original_filename(fl["file_upload"])
            # path to file
            if path_to_files is None:
                # suffix = pl.Path(filename_uploaded).suffix
                # path_to_files = pl.Path(file["data"]["image"])
                file_path = pl.Path(filename)
            else:
                file_path = pl.Path(path_to_files).with_name(filename)

            img_sz = self._extract_original_size(fl)
            yield file_path, img_sz, fl

    # ----- COCO-style data
    def _coco_images(self, path_to_files: Union[str, pl.Path] = None) -> List[dict]:
        coco_images = []
        for nm, img_sz, fl in self._iter_images(path_to_files):
            coco_images.append({"width": img_sz[0],
                                "height": img_sz[1],
                                "id": fl["id"],  # FIXME: what is the id here?
                                "file_name": nm.as_posix()  # TODO: ensure uniqueness of filenames
                                })
        return coco_images

    def _coco_categories(self) -> List[dict]:
        categories, categories_keypoints = [], []
        # label-studio organizes annotations per file => loop over all files
        for fl in self._iterfiles():
            for lbl, ty, val in self._iterlabels(fl):
                if lbl not in categories:
                    if self._istypekeypoint(ty):
                        if lbl not in categories_keypoints:
                            categories_keypoints.append(lbl)
                    else:
                        categories.append(lbl)

        if len(categories) <= 1 and len(categories_keypoints) > 0:
            coco_categories = [{"supercategory": categories[0] if len(categories) == 1 else None,
                                "id": 1 if len(categories) == 1 else None,
                                "name": None,
                                "keypoints": categories_keypoints
                                }]
        else:
            coco_categories = [{"id": i, "name": c} for i, c in enumerate(categories)]
        return coco_categories

    def _coco_annotations(self, coco_categories):
        category_names = [el["supercategory"] for el in coco_categories]
        category_ids = [el["id"] for el in coco_categories]

        coco_annotations = []
        id = 0  # annotation id
        for fl in self._iterfiles():
            image_id = fl["id"]
            # image_id = filenames.index(pl.Path(filenames[file_ids.index(id)]).name)
            keypoint_categories = coco_categories[0]["keypoints"]  # FIXME: indexing

            original_width, original_height = self._extract_original_size(fl)

            keypoints = [0, 0, 0] * len(keypoint_categories)
            bbox, segmentation = [], []  # FIXME: add segmentation
            category_id = -1
            for lbl, ty, val in self._iterlabels(fl):
                if self._istypebbox(ty):
                    category_id = category_ids[category_names.index(lbl)]
                    # Label Studio stores coordinates as relative values
                    bbox_rel = [val[ky] for ky in ["x", "y", "width", "height"]]
                    bbox = self._sacle_to_abs_coordinates(bbox_rel, original_width, original_height)
                    assert bbox <= [original_width, original_height] * 2, "Coordinates outside image size!"
                elif self._istypekeypoint(ty):
                    # keypoints are ordered
                    idx = keypoint_categories.index(lbl) * 3
                    keypoints[idx + 0: idx + 2] = self._sacle_to_abs_coordinates([val["x"], val["y"]],
                                                                                 original_width,
                                                                                 original_height)
                    keypoints[idx + 2] = 2  # FIXME: read visibility from JSON file
                    # v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible
                    assert len(keypoints) == len(keypoint_categories) * 3, "Length of list 'keypoints' changed unexpectedly!"  # FIXME: delete

            coco_annotations.append({"id": id,
                                     "image_id": image_id,
                                     "category_id": category_id,
                                     "segmentation": segmentation,
                                     "bbox": bbox,
                                     "area": self._to_decimal_precision(bbox[2] * bbox[3], 5) if len(bbox) > 0 else 0,
                                     "ignore": 0,
                                     "iscrowd": 0,
                                     "keypoints": keypoints,
                                     "num_keypoints": int(len(keypoints) / 3)
                                     })
            # update annotation id
            id += 1
        return coco_annotations

    def get_coco_data(self, path_to_files: Union[str, pl.Path] = None) -> dict:
        data_coco_images = self._coco_images(path_to_files=path_to_files)
        data_coco_categories = self._coco_categories()
        data_coco_annotations = self._coco_annotations(data_coco_categories)

        # put everything together
        data_coco = {"images": data_coco_images,
                     "categories": data_coco_categories,
                     "annotations": data_coco_annotations,
                     "info": []}

        # # info
        # data_coco["info"].append({"year": datetime.datetime(file["created_at"]).year,  # TODO: test
        #                           "contributor": "Label Studio converter",
        #                           "date_created": file["created_at"]
        #                           })
        return data_coco

    @staticmethod
    def ensure_uniqueness_of_annotations(annotation_data: dict,
                                         keys_with_unique_vales: List[str],
                                         error_info: str = None) -> dict:
        # ensure uniqueness
        info = dict()
        for ky in keys_with_unique_vales:
            keys = []
            for an in annotation_data:
                if an[ky] not in keys:
                    keys.append(an[ky])
            if len(keys) == 1:
                info[ky] = keys[0]
            else:
                msg = f"{len(keys)} keys in annotation. No key '{ky}' found in annotations" \
                      + f" for file {error_info}." if error_info else "."
        return info


def extract_original_filename(filename_with_prefix: str) -> str:
    """
    ignores the preceding hash that Label Studio automatically adds to uploaded files
    :param filename_with_prefix: name of the file in LabelStudio
    :return: original name of the file (without the unique hash attached to before the name by LabelStudio)
    """
    # strip the preceding hash code
    m = re.match("[a-z0-9]+-", filename_with_prefix)
    if m:
        filename = filename_with_prefix[m.end():]
    else:
        filename = filename_with_prefix
    return filename


if __name__ == "__main__":
    path_to_labelstudio = pl.Path("project-1-at-2023-02-14-10-28-8650d03c.json")

    # create class instance and export COCO-style (text) data
    data = LabelStudioAnnotation(path_to_annotations=path_to_labelstudio).get_coco_data()

    # write new file
    with open(path_to_labelstudio.with_name("annotations.json"), "w") as fid:
        json.dump(data, fid)

