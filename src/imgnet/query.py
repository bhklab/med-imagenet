from pydantic import BaseModel, Field


class ValidQuery(BaseModel):
    """
    A valid query to the NBIA API.
    """
    
    collection: str = Field(description="The collection to query", default="all", examples=["all", "4D-Lung", "RADCURE"])


    @field_validator("collection")
    def validate_collection(cls, v: str) -> str:
        if collection in SUPPORTED_COLLECTIONS:
            return collection
        else:
            raise ValueError(f"Collection {collection} not found")
