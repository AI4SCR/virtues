from ai4bmr_core.utils.stats import StatsRecorder
import numpy as np
from pathlib import Path
from external.virtues.utils.transform_utils import CustomGaussianBlur

images_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/virtues/datasets/PCa/images')
stats_recorder = StatsRecorder()

preprocess_name = 'clip_log1p_gaussian_blur'
def process_image(image):
    image = np.clip(image, 0, np.percentile(image, 99, axis=(1, 2), keepdims=True)).astype(np.float32)
    image = np.log1p(image)  # log1p = log(1 + x)
    image = CustomGaussianBlur(kernel_size=3, sigma=1.0)(image)
    return image

for img_path in images_dir.glob('*.npy'):
    img = np.load(img_path)
    img = process_image(img)
    stats_recorder.update(img)
    assert np.isfinite(stats_recorder.mean).all()
    assert np.isfinite(stats_recorder.std).all()

import yaml
stats = {
    'mean': stats_recorder.mean.tolist(),
    'std': stats_recorder.std.tolist()
}
with open(images_dir / f'stats_{preprocess_name}.yaml', 'w') as f:
    yaml.dump(stats, f)
