"""Export worst errors and a montage of MIL patch scores (not localization truth).

python analysis/inspect_mil_predictions.py --predictions evaluation/Celeb-DF-v2_predictions.npz --rgb_dir /path/to/rgb
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--rgb_dir', default='.')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    source = Path(args.predictions)
    out = Path(args.output_dir) if args.output_dir else source.parent / (source.stem + '_inspection')
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        prob, label, names = data['prob'], data['label'], data['image_names']
        error = np.abs(prob - label)
        with (out / 'worst_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image', 'label', 'fusion_prob', 'cls_only_prob', 'mil_prob', 'absolute_error'])
            for i in np.argsort(-error)[:100]:
                writer.writerow([names[i], label[i], prob[i],
                    data['cls_only_prob'][i] if 'cls_only_prob' in data else '',
                    data['mil_prob'][i] if 'mil_prob' in data and len(data['mil_prob']) == len(prob) else '', error[i]])
        maps, paths = data['patch_prob'], data['patch_image_names']
        if len(maps):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from PIL import Image
            count = min(12, len(maps))
            fig, axes = plt.subplots(count, 2, figsize=(7, 3 * count), squeeze=False)
            for i in range(count):
                path = Path(str(paths[i]).replace('\\', '/'))
                if not path.is_absolute():
                    path = Path(args.rgb_dir) / path
                if path.exists():
                    with Image.open(path) as im:
                        axes[i, 0].imshow(im.convert('RGB').resize((224, 224)))
                else:
                    axes[i, 0].text(.05, .5, 'Source image unavailable', wrap=True)
                axes[i, 0].set_title(str(paths[i])[-65:], fontsize=8)
                axes[i, 0].axis('off')
                plot = axes[i, 1].imshow(maps[i], vmin=0, vmax=1, cmap='magma')
                axes[i, 1].set_title('MIL patch probability (weak supervision)')
                fig.colorbar(plot, ax=axes[i, 1])
            fig.tight_layout()
            fig.savefig(out / 'patch_diagnostics.png', dpi=120)
            plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
