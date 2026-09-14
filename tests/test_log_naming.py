import datetime
import pytest
from training.trainer.trainer import get_vietnam_time_str, get_run_name, Trainer
from training.train import parser, init_seed


def test_vietnam_time_str():
    time_str = get_vietnam_time_str()
    # Expect format HHhMM (e.g. 12h55)
    assert len(time_str) == 5
    assert time_str[2] == 'h'
    hours = int(time_str[:2])
    minutes = int(time_str[3:])
    assert 0 <= hours <= 23
    assert 0 <= minutes <= 59

    # Verify that it corresponds to UTC + 7
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    vn_now = utc_now + datetime.timedelta(hours=7)
    expected_hours = vn_now.hour
    assert hours == expected_hours


def test_get_run_name_standard():
    config = {
        'model_name': 'ln_sspanet_mil',
        'manualSeed': 1024,
    }
    run_name = get_run_name(config, time_now='12h55')
    assert run_name == 'ln_sspanet_mil_1024_12h55'


def test_get_run_name_with_task_target():
    config = {
        'model_name': 'ln_sspanet_mil',
        'task_target': 'A_ln_ce',
        'manualSeed': 42,
    }
    run_name = get_run_name(config, time_now='14h30')
    assert run_name == 'ln_sspanet_mil_A_ln_ce_42_14h30'


def test_get_run_name_with_seed_key():
    config = {
        'model_name': 'camil',
        'seed': 2024,
    }
    run_name = get_run_name(config, time_now='09h15')
    assert run_name == 'camil_2024_09h15'


def test_get_run_name_backward_compat_smoke():
    config = {
        'model_name': 'ln_sspanet_mil',
        'manualSeed': 1024,
    }
    run_name = get_run_name(config, time_now='smoke')
    assert run_name == 'ln_sspanet_mil_smoke'

    config_bias = {
        'model_name': 'bias_sspanet_mil',
    }
    run_name_bias = get_run_name(config_bias, time_now='smoke_bias')
    assert run_name_bias == 'bias_sspanet_mil_smoke_bias'


def test_get_run_name_live_vietnam_time():
    config = {
        'model_name': 'ln_sspanet_mil',
        'manualSeed': 1024,
    }
    run_name = get_run_name(config)
    vn_time = get_vietnam_time_str()
    assert run_name == f"ln_sspanet_mil_1024_{vn_time}"


def test_parser_seed_arguments():
    args1 = parser.parse_args(['--seed', '42'])
    assert args1.seed == 42

    args2 = parser.parse_args(['--manualSeed', '100'])
    assert args2.seed == 100


def test_init_seed_resolution():
    config = {'manualSeed': None, 'cuda': False}
    init_seed(config)
    assert isinstance(config['manualSeed'], int)
    assert 1 <= config['manualSeed'] <= 10000
