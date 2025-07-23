from pydantic import BaseModel, Field, field_validator
from imgtools.dicom import Interlacer
from dataclasses import dataclass
from pathlib import Path
import re
import json
from rich import print 

ROOT_DIR = Path("indexed_datasets")
SUPPORTED_COLLECTIONS = ["4D-Lung", "Adrenal-ACC-Ki67-Seg"]

@dataclass
class Rule:
    tag: str
    value: str | list[str]
    def evaluate(self, dicom_element: dict)->bool:
        tag_values = dicom_element[self.tag]
        if isinstance(tag_values, str):
            tag_values = [tag_values]
        patterns = self.value
        if isinstance(self.value, str):
            patterns = [self.value]
        for tag_value in tag_values:
            for pattern in patterns:
                if re.search(pattern, tag_value) is not None:
                    return True
        return False

class ValidQuery(BaseModel):
    collections: str | list[str] = Field(
        description="The collections to query", 
        default="all", 
        examples=[
            "all", 
            "4D-Lung", 
            ["4D-Lung", "RADCURE"]])
    modalities: str | list[str] = Field(
        description="The modalities to query", 
        default="all", 
        examples=[
            "all", 
            "MR,RTSTRUCT", 
            ["MR,RTSTRUCT", "CT,RTSTRUCT"]])
    rules: dict[str, Rule | list[Rule]] = Field(
        description="The query filter rules, optionally grouped by modality", 
        default=None, 
        examples=[
            {
                "RTSTRUCT": "ROINames == ['lung', 'lung*']",
                "MR": ["ImageType=='PRIMARY'", 
                       "PixelPaddingValue == 1"]
                }])

    @field_validator("collections", mode="after")
    def validate_collections(cls, v: str | list[str]
                            ) -> str | list[str]:
        """
        Pydantic validation of collection field.
        Checks if the collections supplied are supported by med-imagenet.
        """
        if isinstance(v, list):
            for _collection in v:
                if _collection not in SUPPORTED_COLLECTIONS:
                    raise ValueError(f"Collection {_collection} not found.")
        else:
            if v != "all" and v not in SUPPORTED_COLLECTIONS:
                raise ValueError(f"Collection {v} not found.")
        return v
    @field_validator("modalities", mode="after")
    def validate_modalities(cls, v: str | list[str]
                            ) -> str | list[str]:
        return v
    
    @field_validator("rules", mode="before")
    def validate_rules(cls, v: any
                       ) -> dict[str, Rule | list[Rule]]:
        def parse_rule(rule:str)->Rule:
            """What the hell was i meaning to do here?
            So I guess parse the rule and figure out what the dicom tag u need to access is, 
            figure out the comparison type, figure out if the value is a list or not? """

            rule_parts = rule.split(" ", 2)
            tag = rule_parts[0]
            value = rule_parts[2]
            if value[0] == '[':
                # there is a list of patterns instead of just one.
                # Using regex to parse the list and get each individual element.
                matches = re.findall(r'''(['"])(.*?)\1''', value)
                value = [m[1] for m in matches]
            else: 
                value = value.strip("\'\"")
            return Rule(tag=tag, value=value)
        return_value = {}
        # Check if v is of type dict[str, str]
        if (
            isinstance(v, dict) 
                and all(isinstance(key, str) 
                    and isinstance(val, str) 
                for key, val in v.items())
            ):
            
            for key in v: 
                return_value[key] = parse_rule(v[key])
        # Check if v is of type dict[str, list[str]]
        elif (
             isinstance(v, dict)
                and all(isinstance(key, str) 
                        and isinstance(val, list) 
                        and all(isinstance(list_val, str) 
                            for list_val in val) 
                    for key, val in v.items())
             ):
            for key in v:
                for i in range(len(v[key])):
                    return_value[key][i] = parse_rule(v[key][i])
        else:
            # v may already be in its parsed form, so we set return_value to v and check if its correct in the next block.
            return_value = v
        # Check if return_value is of type dict[str, Rule|list[Rule]]
        # if return value is a dict, and that dict contains keys of type string and the values are either Rule or list[Rule]
        if (
            isinstance(return_value, dict)
                and (all(isinstance(key, str) 
                         and (isinstance(val, Rule)
                              or isinstance(val, list) 
                                and all(isinstance(list_val, Rule) 
                                for list_val in val)))
                         for key, val in return_value.items())
           ):
            return return_value
        else: 
            raise ValueError(f"Invalid rules: {v}.")

def process_query(query: ValidQuery) -> dict[str, list[str]]:
    """Given a ValidQuery, returns a dictionary file containing the seriesUID for each series
    matching the query by collection."""
    collections = query.collections
    modalities = query.modalities
    rules = query.rules

    matches = {}
    if collections == "all":
        collections = SUPPORTED_COLLECTIONS
    if isinstance(collections, str):
        collections = [collections]
    if isinstance(modalities, str):
        modality_queries = [modalities]
    
    for collection in collections:
        # Get index csv
        csv_path = ROOT_DIR / ".imgtools"/ collection / "index.csv"

        # Access crawl json
        with open(ROOT_DIR / ".imgtools" / collection / "crawl_db.json", "r") as f:
            crawl_db = json.load(f)
        interlacer = Interlacer(csv_path)
        modality_matches = []
        
        for query in modality_queries:
            if query == 'all':
                query_result = interlacer.query_all()
            else:
                query_result = interlacer.query(query)
            modality_matches += [node.SeriesInstanceUID for group in query_result for node in group]
        result = []
        for series in modality_matches:

            for key in crawl_db[series]:
                # The crawldb is structured weird, there's always an extra layer in between
                # the serieduid and the actual metadata associated with it.
                # This is just a easy workaround.
                dicom = crawl_db[series][key]
            
            modality = dicom['Modality']
            modality_rules = rules.get(modality)
            if isinstance(modality_rules, Rule):
                modality_rules = [rules[modality]]
            
            accept_series = True
            if modality_rules is not None:
                for rule in modality_rules:
                    if not rule.evaluate(dicom):
                        # a series is only added to the final query output if it follows ALL the rules.
                        accept_series = False
                        break
            if accept_series:
                result.append(series)
        matches[collection] = result
    return matches
                

query = ValidQuery(collections="all", modalities="CT,RTSTRUCT", rules={"RTSTRUCT": "ROINames == ['lung.*', 'lung']"})

result = process_query(query)

for key in result:
    print(key)
    for item in result[key]:
        print(f"    {item}")




                





                


