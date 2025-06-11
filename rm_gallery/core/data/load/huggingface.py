from typing import List

from rm_gallery.core.data.load.base import DataLoad, DataLoadStrategyRegistry
from rm_gallery.core.data.schema import DataSample


@DataLoadStrategyRegistry.register("remote", "huggingface")
class HuggingfaceDataLoadStrategy(DataLoad):
    def load_data(self, **kwargs) -> List[DataSample]:
        pass
