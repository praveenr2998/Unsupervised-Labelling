from pydantic import BaseModel
from typing import List, Union, Optional

class LabellingResponseSubModel(BaseModel):
    id: str
    label: str

class LabellingResponseModel(BaseModel):
    labelled_data: List[LabellingResponseSubModel]