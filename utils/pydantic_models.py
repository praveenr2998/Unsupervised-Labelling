from pydantic import BaseModel
from typing import List, Union, Optional

class LabellingResponseSubModel(BaseModel):
    id: int
    label: str

class PoorQualityLabellingResponseModel(BaseModel):
    labelled_data: List[LabellingResponseSubModel]

class GoodQualityLabellingResponseModel(BaseModel):
    label: str