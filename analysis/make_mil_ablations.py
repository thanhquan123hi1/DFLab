"""Generate full detector configs with only architecture/MIL switches changed."""
import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='training/config/detector/ln_sspanet_mil.yaml')
    parser.add_argument('--output_dir', default='training/config/detector/ablations')
    args = parser.parse_args()
    with open(args.base, encoding='utf-8') as stream:
        base = yaml.safe_load(stream)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = {'A_ln_ce': (False, False, 0.), 'B_ln_patch_mil': (True, False, .3),
            'C_ln_sspa_ce': (True, True, 0.), 'D_ln_sspa_mil': (True, True, .3)}
    for name, (patch, sspa, weight) in runs.items():
        config = dict(base, use_patch=patch, use_sspanet=sspa, lambda_mil=weight, task_target=name)
        with (output / (name + '.yaml')).open('w', encoding='utf-8') as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    print(output)


if __name__ == '__main__':
    main()
