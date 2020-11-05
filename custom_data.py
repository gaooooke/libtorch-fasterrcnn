from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import register_pascal_voc
from detectron2.data import DatasetCatalog, MetadataCatalog
import os

# coco
# register_coco_instances("custom", {}, "./COCO/trainval.json", "./COCO/images")
# custom_metadata = MetadataCatalog.get("custom")

# voc

# CLASS_NAMES=('headset','mask','hands','gloves','shoes','nomask','noheadset',)
# CLASS_NAMES = ('fire',)
CLASS_NAMES = ("hair","nohair","clothes","nohat",)

SPLITS = [
        ("custom", "hair", "train"),
        # ("custom_val","hair","val"),
        # ("voc_self_val", "VOC2007", "trainval"),
    ]

for name, dirname, split in SPLITS:
    year = 2007 if "2007" in name else 2012
    register_pascal_voc(name, os.path.join("../", dirname), split, None, class_names=CLASS_NAMES)
    MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    # MetadataCatalog.get(name).thing_classes = ["clothes"]
