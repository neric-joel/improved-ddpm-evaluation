"""Convert the downloaded parquet files (Hugging Face mirrors of CIFAR-10 and SVHN) into uint8 .npy arrays."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from common import DATA


def convert(src, dst, img_col, label_col):
    table = pq.read_table(src)
    cols = table.column_names
    img_col = img_col if img_col in cols else [c for c in cols if "im" in c][0]
    label_col = label_col if label_col in cols else [c for c in cols if "label" in c][0]
    imgs = table.column(img_col).to_pylist()
    labels = np.array(table.column(label_col).to_pylist(), dtype=np.int64)
    arr = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in imgs]).astype(np.uint8)
    np.save(DATA / f"{dst}.npy", arr)
    np.save(DATA / f"{dst}_labels.npy", labels)
    print(dst, arr.shape, arr.dtype, "labels", np.bincount(labels))


if __name__ == "__main__":
    convert(DATA / "cifar10" / "train.parquet", "cifar_train", "img", "label")
    convert(DATA / "cifar10" / "test.parquet", "cifar_test", "img", "label")
    convert(DATA / "svhn" / "test.parquet", "svhn_test", "image", "label")
