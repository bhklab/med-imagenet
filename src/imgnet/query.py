from pydantic import BaseModel, Field, field_validator
from imgtools.dicom import Interlacer
from dataclasses import dataclass
from pathlib import Path
import re
import json
from rich import print 
from enum import Enum


ROOT_DIR = Path("indexed_datasets")
SUPPORTED_COLLECTIONS = ["4D-Lung", "Adrenal-ACC-Ki67-Seg", "C4KC-KiTS"]

class RuleError(Exception):
    """Exception raised for invalid rules.

    Attributes
    ----------
        message: str 
            explanation of the error.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

@dataclass
class Rule:
    tag: str
    value: str | list[str]
    comparison: str
    def evaluate(self, dicom_element: dict)->bool:
        tag_value = dicom_element.get(self.tag)
        if isinstance(tag_value, str):
            # If it starts with "[" and ends with "]", treat as a list
            if tag_value.strip().startswith('[') and tag_value.strip().endswith(']'):
                # Extract quoted and unquoted items
                matches = re.findall(r'''(['"])(.*?)\1|([^'",\s\[\]]+)''', tag_value)
                tag_value = [m[1] if m[1] else m[2] for m in matches]
            else:
                # Treat the entire string as a single item
                tag_value = [tag_value.strip()]
        match self.comparison:
            case "==" | "=":
                patterns = self.value
                if isinstance(self.value, str):
                    patterns = [self.value]
                for element in tag_value:
                    for pattern in patterns:
                        if re.search(pattern, element):
                            return True
                return False
            case ">":
                for element in tag_value:
                    if element == "" or element is None:
                        return False
                    try:
                        comparison_value = float(self.value)
                        element = float(element)
                        if element <= comparison_value:
                            return False
                    except ValueError:
                        raise RuleError("'>' comparisons only support numeric values."
                                        +f"\nInput: {self.tag}: {tag_value}, > {self.value}")
                return True
            case "<":
                for element in tag_value:
                    if element == "" or element is None:
                        return False
                    try:
                        comparison_value = float(self.value)
                        element = float(element)
                        if element >= comparison_value:
                            return False
                    except ValueError:
                        raise RuleError("'>' comparisons only support numeric values."
                                        +f"\nInput: {self.tag}: {tag_value}, > {self.value}")
                return True
            case ">=":
                for element in tag_value:
                    if element == "" or element is None:
                        return False
                    try:
                        comparison_value = float(self.value)
                        element = float(element)
                        if element < comparison_value:
                            return False
                    except ValueError:
                        raise RuleError("'>' comparisons only support numeric values."
                                        +f"\nInput: {self.tag}: {tag_value}, > {self.value}")
                return True
            case "<=":
                for element in tag_value:
                    if element == "" or element is None:
                        return False
                    try:
                        comparison_value = float(self.value)
                        element = float(element)
                        if element > comparison_value:
                            return False
                    except ValueError:
                        raise RuleError("'>' comparisons only support numeric values."
                                        +f"\nInput: {self.tag}: {tag_value}, > {self.value}")
                return True
            case "!=":
                patterns = self.value
                if isinstance(self.value, str):
                    patterns = [self.value]
                for element in tag_value:
                    for pattern in patterns:
                        if re.search(pattern, element):
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
                "RTSTRUCT": "ROINames == ['lung', 'lung.*']",
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
            comparison = rule_parts[1]
            if comparison not in ["=", "==", "<", ">", "<=", ">=", "!="]:
                raise ValueError(f"{comparison} is not a supported comparison type."
                                 +"\n supported comparison types: ==, <, >, <=, >=, !=")

            value = rule_parts[2]
            if value[0] == '[':
                # there is a list of patterns instead of just one.
                # Using regex to parse the list and get each individual element.
                matches = re.findall(r'''(['"])(.*?)\1''', value)
                value = [m[1] for m in matches]
            else: 
                value = value.strip("\'\"")
            return Rule(tag=tag, value=value, comparison=comparison)
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

    def process(self) -> dict[str, list[str]]:
        """Given a ValidQuery, returns a dictionary file containing the seriesUID for each series
        matching the query by collection."""
        collections = self.collections
        modality_queries = self.modalities
        rules = self.rules

        matches = {}
        if collections == "all":
            collections = SUPPORTED_COLLECTIONS
        if isinstance(collections, str):
            collections = [collections]
        if isinstance(modality_queries, str):
            modality_queries = [modality_queries]
        
        for collection in collections:
            print(collection)
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
                modality_matches += [[node.SeriesInstanceUID for node in group] for group in query_result]
            result = []
            for group in modality_matches:
                for series in group:
                    
                    for key in crawl_db[series]:
                        # The crawldb is structured weird, there's always an extra layer in between
                        # the seriesuid and the actual metadata associated with it.
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
                    elif series == group[0]:
                        # if the root node (the first element in the list) is not selected by the query, the children will be skipped.
                        # this prevents the selection of masks which reference a dicom that was not selected.
                        break 
            matches[collection] = result
        return matches
                    

if __name__ == "__main__":
    query = ValidQuery(
                collections="all", 
                modalities=["CT,SEG", "CT,RTSTRUCT"], 
                rules={
                    "CT": "PixelSpacing < 0.9",
                    "SEG": "ROINames == 'Mass'",
                    "RTSTRUCT": "ROINames == ['Lung']"
                    })



    result = query.process()

    for key in result:
        print(key)
        for item in result[key]:
            print(f"    {item}")




                

"""
query
- add < and > 
- fix imgnet.py query function
- pytests for query
- add cli entrypoint # UGHHH I HAVE TO SET UP CLICK
- pydantic json schema stuff (just save json and schema in users output dir)
- add downloading of queried stuff # i think josh is doing ts?
index all datasets
- 

Josh vacay in 2 weeks

"""



                


