from pathlib import Path
from jsonargparse import CLI
import torch

def main(embeddings_dir: Path):
    for embed_path in embeddings_dir.glob('*.pt'):
        embed = torch.load(embed_path)
        embed = embed['mean_representations'][30]
        torch.save(embed, embed_path)


if __name__ == '__main__':
    CLI(main, as_positional=False)