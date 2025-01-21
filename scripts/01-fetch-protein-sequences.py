from pathlib import Path
from jsonargparse import CLI
import pandas as pd
import requests
import re

def main(dataset_dir: Path = Path('/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/virtues/datasets/PCa')):
    save_path = dataset_dir / 'protein-sequences.fasta'
    panel = pd.read_csv(dataset_dir / 'panel.csv')

    url = 'https://www.uniprot.org/uniprotkb'
    format = 'fasta'
    resps = []
    for uniport_id in panel.uniprot_id:
        url_ = f'{url}/{uniport_id}.{format}'
        resp = requests.get(url_)

        if resp.status_code == 200:
            txt = resp.text
            txt_with_only_uniprot_id = re.sub(r'>sp\|([^|]+)\|.*', r'>\1', txt)
            resps.append(txt_with_only_uniprot_id)
        else:
            print(f"Failed with status code: {resp.status_code}")
            print(resp.text)

    with open(save_path, 'w') as f:
        f.writelines(resps)


if __name__ == "__main__":
    CLI(main, as_positional=False)