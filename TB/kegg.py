from Bio.KEGG import REST


def get(ID="K18766", db_id=None):
    """
    Fetch data from the KEGG database.
    -----
    Args:
    ID: str
    db_id: str, default None
        - 'ko' : request to onthology database
        - 'ec' : request to enzyme database
        - 'cpd': request to compound database
        - 'rn' : request to reaction database
    If db_id is None the request will be directed
    based on the first letter of the ID.
      C -> compound
      K -> orthology
      R -> reaction
    all other requests will be directed to the enzyme database
    """

    db_keys = {"orthology": "ko", "enzyme": "ec", "compound": "cpd", "reaction": "rn"}
    if db_id is not None:
        db_key = db_keys[db_id]
    elif ID.startswith("C"):
        db_key = db_keys["compound"]
    elif ID.startswith("K"):
        db_key = db_keys["orthology"]
    elif ID.startswith("R"):
        db_key = db_keys["reaction"]
    else:
        db_key = db_keys["enzyme"]

    data = REST.kegg_get(f"{db_key}:{ID}").read().split("\n")

    return data
