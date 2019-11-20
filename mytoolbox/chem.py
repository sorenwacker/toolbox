import requests

def get_smiles_from_inchikey(inchikey):
    r = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSMILES/JSON').json()
    return r['PropertyTable']['Properties'][0]['CanonicalSMILES']