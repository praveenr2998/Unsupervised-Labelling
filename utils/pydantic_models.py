from typing import List, Optional, Union

from pydantic import BaseModel


class LabellingResponseSubModel(BaseModel):
    id: int
    label: str


class PoorQualityLabellingResponseModel(BaseModel):
    labelled_data: List[LabellingResponseSubModel]


class GoodQualityLabellingResponseModel(BaseModel):
    label: str
