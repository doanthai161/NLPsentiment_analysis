from pydantic import BaseModel

class TextBatchData(BaseModel):
    texts: list[str]

class TextData(BaseModel):
    text: str

