from pydantic import BaseModel, Field, field_validator
from imgtools.dicom import Interlacer
from dataclasses import dataclass
from pathlib import Path
import re
import json
from rich import print 
from enum import Enum
import pandas as pd


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

class InvalidComparisonError(RuleError):
    """Exception raised when a Rule has an invalid comparison type for the given argument type."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ValidQueryError(Exception):
    """BaseException for ValidQuery errors."""
    pass
class ModalitiesValidationError(ValidQueryError):
    """Exception raised when modality field validation fails."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
class CollectionsValidationError(ValidQueryError):
    """Exception raised when collections field validation fails."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
class RulesValidationError(ValidQueryError):
    """Exception raised when rules field validation fails."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class RulesValidationParsingError(RulesValidationError):
    """Exception raised when parsing a Rule from string fails during rules field validation."""
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
        if tag_value is None:
            return False
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
                        if re.match(pattern, element):
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
                        if re.match(pattern, element):
                            return False
                return True
                


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
    rules: dict[str, Rule | list[Rule]] | None = Field(
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
                    raise CollectionsValidationError(f"Collection {_collection} not found.")
        else:
            if v != "all" and v not in SUPPORTED_COLLECTIONS:
                raise CollectionsValidationError(f"Collection {v} not found.")
        return v
    @field_validator("modalities", mode="before")
    def validate_modalities(cls, v: any
                            ) -> str | list[str]:
        if isinstance(v, str):
            return v
        elif isinstance(v, list) and all(isinstance(val, str) for val in v):
            return v
        raise ModalitiesValidationError(f"modalities must be of type str | list[str]. Got type {type(v)} instead.")
    
    @field_validator("rules", mode="before")
    def validate_rules(cls, v: any
                       )-> dict[str, Rule | list[Rule]]:
        if v is None:
            return None
        def parse_rule(rule:str)->Rule:
            """What the hell was i meaning to do here?
            So I guess parse the rule and figure out what the dicom tag u need to access is, 
            figure out the comparison type, figure out if the value is a list or not? """

            rule_parts = rule.split(" ", 2)
            if len(rule_parts) != 3:
                raise RulesValidationParsingError("Invalid rule syntax.")
            tag = rule_parts[0]
            comparison = rule_parts[1]
            if comparison not in ["=", "==", "<", ">", "<=", ">=", "!="]:
                raise RulesValidationParsingError(f"{comparison} is not a supported comparison type."
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
        if isinstance(v, dict):
            for key, val in v.items():
                if isinstance(key, str):
                    if isinstance(val, str):
                        return_value[key] = parse_rule(val)
                    # check if val is a list[str]
                    elif isinstance(val, list) and all(isinstance(list_val, str) for list_val in val):
                        return_value[key] = [parse_rule(list_val) for list_val in val]
                    # check if val is a Rule or list[Rule]
                    elif isinstance(val, Rule) or (isinstance(val, list) and all(isinstance(list_val, Rule) for list_val in val)):
                        return_value[key] = val
                    else:
                        raise RulesValidationError(f"rules must be dict[str, str | list[str]], got dict[str, str | list[str] | {type(val)}] instead.")
                else: 
                    raise RulesValidationError(f"rules must be dict[str, str | list[str]], got dict[{type(key)}, str | list[str]] instead.")
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
            raise RulesValidationError(f"Invalid rules: {v}.")

    def process(self) -> pd.DataFrame:
        """Given a ValidQuery, returns a dictionary file containing the seriesUID for each series
        matching the query by collection."""
        collections = self.collections
        modality_queries = self.modalities
        rules = self.rules

        matches = []
        if collections == "all":
            collections = SUPPORTED_COLLECTIONS
        if isinstance(collections, str):
            collections = [collections]
        if isinstance(modality_queries, str):
            modality_queries = [modality_queries]
        
        for collection in collections:
            # Get index csv
            csv_path = ROOT_DIR / ".imgtools"/ collection / "index.csv"

            # Access crawl json
            with open(ROOT_DIR / ".imgtools" / collection / "crawl_db.json", "r") as f:
                crawl_db = json.load(f)
            interlacer = Interlacer(csv_path)
            modality_matches = []
            
            for query in modality_queries:
                if query == 'all' or query is None:
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
                    accept_series = True
                    if rules:
                        modality_rules = rules.get(modality)
                        if isinstance(modality_rules, Rule):
                            modality_rules = [rules[modality]]
                        if modality_rules:
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
            index_df = pd.read_csv(csv_path)
            index_df = index_df[index_df["SeriesInstanceUID"].isin(result)]
            index_df["Collection"] = collection
            matches.append(index_df)
        return pd.concat(matches, ignore_index=True)
                    

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

    print(result)




                

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



                


