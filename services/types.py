from plistlib import Dict

import numpy as np


# data type for predictions with 'boxes', 'class_ids', 'scores', 'masks'
def Prediction(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {"boxes": boxes, "class_ids": class_ids, "scores": scores, "masks": masks}
