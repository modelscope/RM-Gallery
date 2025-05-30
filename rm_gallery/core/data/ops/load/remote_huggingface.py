from typing import List

from rm_gallery.core.data.load import DataLoadStrategy, DataLoadStrategyRegistry
from rm_gallery.core.data.schema import DataSample


@DataLoadStrategyRegistry.register("remote", "huggingface", "*")
class HuggingfaceDataLoadStrategy(DataLoadStrategy):
    def load_data(self, **kwargs) -> List[DataSample]:
        pass
