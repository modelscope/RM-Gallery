from rm_gallery.gallery.data.annotation.rewardbench_template import (
    RewardBenchAnnotationTemplate,
)
from rm_gallery.gallery.data.load.local_helpsteer2 import HelpSteer2DataLoadStrategy
from rm_gallery.gallery.data.load.local_prmbench import PRMDataLoadStrategy
from rm_gallery.gallery.data.load.local_rewardbench import RewardBenchDataLoadStrategy
from rm_gallery.gallery.data.load.local_rewardbench_2 import (
    RewardBench2DataLoadStrategy,
)
from rm_gallery.gallery.data.load.local_rmbbenchmark_bestofn import (
    RMBBenchmarkBestOfNDataLoadStrategy,
)
from rm_gallery.gallery.data.load.local_rmbbenchmark_pairwise import (
    RMBBenchmarkPairwiseDataLoadStrategy,
)

LOAD_STRATEGIES = {
    "rewardbench": RewardBenchDataLoadStrategy,
    "prmbench": PRMDataLoadStrategy,
    "helpsteer2": HelpSteer2DataLoadStrategy,
    "rewardbench-2": RewardBench2DataLoadStrategy,
    "rmbbenchmark_bestofn": RMBBenchmarkBestOfNDataLoadStrategy,
    "rmbbenchmark_pairwise": RMBBenchmarkPairwiseDataLoadStrategy,
}

ANNOTATION_TEMPLATES = {
    "rewardbench": RewardBenchAnnotationTemplate,
}
