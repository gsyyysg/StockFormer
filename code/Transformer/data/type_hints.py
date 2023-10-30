from __future__ import annotations

from typing import Tuple, List, Union

import numpy as np

FeatureType = Tuple[float, float, float, float, float]  # (open, close, high, low, volume)
StockFeatureList = List  # Feature of a stock
StockFeatureData = List[StockFeatureList[FeatureType]]
ContinuousFeaturesOfOneStock = Union[StockFeatureList[FeatureType], np.ndarray]