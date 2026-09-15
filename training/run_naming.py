"""Shared training directory names using Vietnam local time."""
from datetime import datetime, timedelta, timezone


def get_run_name(config, time_now=None):
    stamp = time_now or datetime.now(timezone(timedelta(hours=7))).strftime('%Hh%M')
    seed = config.get('manualSeed')
    seed_part = f'_{seed}' if seed is not None else ''
    return f"{config['model_name']}{seed_part}_{stamp}"
