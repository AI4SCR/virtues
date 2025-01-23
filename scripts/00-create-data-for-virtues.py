from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import CLI
from skimage.io import imread

from ai4bmr_core.utils.logging import get_logger


def main(
        dataset_dir: Path,
        save_dir: Path,
        name: str = 'PCa',
        sample_names: list[str] | None = None,
        verbose: int = 0
):
    logger = get_logger('create-virtues-data', verbose=verbose)
    logger.info(f'create datasets `{name}` from {dataset_dir}')

    # %%
    dataset_dir = dataset_dir.expanduser().resolve()
    save_dir = save_dir.expanduser().resolve()

    # %%
    embed_save_dir = save_dir / 'embeddings'
    embed_save_dir.mkdir(parents=True, exist_ok=True)

    dataset_save_dir = Path(save_dir / 'datasets' / name)
    img_save_dir = dataset_save_dir / 'images'
    img_save_dir.mkdir(parents=True, exist_ok=True)

    mask_save_dir = dataset_save_dir / 'masks'
    mask_save_dir.mkdir(parents=True, exist_ok=True)

    panel_save_path = dataset_save_dir / 'panel.csv'
    panel_save_path.parent.mkdir(parents=True, exist_ok=True)

    clinical_save_path = dataset_save_dir / 'clinical.csv'
    sce_annotations_path = dataset_save_dir / 'sce_annotations.csv'

    metadata_save_path = save_dir / 'metadata' / name / f'gene_dict_{name}.csv'
    metadata_save_path.parent.mkdir(parents=True, exist_ok=True)

    # panel
    panel_path = dataset_dir / 'images' / 'filtered' / 'panel.csv'
    panel = pd.read_csv(panel_path)
    filter_channels = panel.uniprot_id.notna()
    panel = panel[filter_channels]
    panel.to_csv(panel_save_path, index=False)

    # clinical
    clinical_path = dataset_dir / 'metadata' / '02_processed' / 'clinical_metadata.parquet'
    clinical = pd.read_parquet(clinical_path)
    clinical = clinical.reset_index()

    # labels
    labels_path = dataset_dir / 'metadata' / '02_processed' / 'labels.parquet'
    labels = pd.read_parquet(labels_path)
    labels = labels.reset_index()

    # images and masks
    images_dir = dataset_dir / 'images' / 'filtered'
    masks_dir = dataset_dir / 'masks' / 'cleaned'
    sample_names = sample_names or set(map(lambda x: x.stem, images_dir.glob('*.tiff')))

    # note: ensure we only have samples with annotations
    sample_names = sample_names.intersection(set(labels.sample_name))
    sample_names = sample_names.intersection(set(clinical.sample_name))
    sample_names = list(sample_names)

    logger.info(f'dataset has {len(sample_names)} samples')

    for i, sample_name in enumerate(sample_names, start=1):
        logger.info(f'processing sample_name: [{i}/{len(sample_names)}] {sample_name}')
        img_path = images_dir / f'{sample_name}.tiff'
        img = imread(img_path)[filter_channels]
        image_name = f'{sample_name}.npy'
        np.save(str(img_save_dir / image_name), img)

        mask_path = masks_dir / f'{sample_name}.tiff'
        mask = imread(mask_path)
        np.save(str(mask_save_dir / image_name), mask)

    # %%

    # clinical
    clinical = clinical.rename(columns={'pat_id': 'patient_id'})
    clinical = clinical.assign(image_name=clinical.sample_name)

    clinical_cols = ['image_name', 'patient_id', 'psa_progr', 'disease_progr', 'os_status']
    clinical = clinical[clinical_cols]
    filter_ = clinical.image_name.isin(sample_names)
    clinical = clinical[filter_]
    assert len(clinical) == len(sample_names)
    clinical.to_csv(clinical_save_path, index=False)

    # sce_annotations
    sce_annotations = labels.assign(image_name=labels.sample_name, cell_id=labels.object_id)
    sce_anno_cols = ['image_name', 'cell_id', 'main_group', 'meta_group', 'label']

    sce_annotations = sce_annotations[sce_anno_cols]
    filter_ = sce_annotations.image_name.isin(sample_names)
    sce_annotations = sce_annotations[filter_]
    sce_annotations.to_csv(sce_annotations_path, index=False)

    # metadata
    metadata = panel[['name', 'uniprot_id']].rename(columns=dict(uniprot_id='protein_id'))
    metadata.to_csv(metadata_save_path, index=False)


if __name__ == "__main__":
    CLI(main, as_positional=False)
