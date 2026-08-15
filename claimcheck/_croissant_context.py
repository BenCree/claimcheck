"""The official Croissant 1.1 `@context`, copied verbatim.

Not hand-rolled. A minimal context of our own DID validate — because
`mlcroissant` could not resolve `field` against it and therefore checked
nothing. With the real one it immediately found a required property
missing from every field. A validator that cannot resolve your terms is
not validating, and a clean report from one is worth nothing.
"""

CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs", "column": "cr:column",
    "conformsTo": "dct:conformsTo", "containedIn": "cr:containedIn",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract", "field": "cr:field",
    "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet", "format": "cr:format", "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath",
    "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField",
    "path": "cr:path", "prov": "http://www.w3.org/ns/prov#",
    "recordSet": "cr:recordSet", "references": "cr:references",
    "regex": "cr:regex", "repeated": "cr:repeated", "replace": "cr:replace",
    "sc": "https://schema.org/", "samplingRate": "cr:samplingRate",
    "separator": "cr:separator", "source": "cr:source",
    "subField": "cr:subField", "transform": "cr:transform",
}
