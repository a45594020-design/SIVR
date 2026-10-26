
import os
from datetime import datetime

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # else:
    #     shutil.rmtree(dir_name, ignore_errors=True)
    #     os.makedirs(dir_name)


def get_str_experiments(flags):
    dateTimeObj = datetime.now()
    dateStr = dateTimeObj.strftime("%Y_%m_%d")
    parts = [flags.dataset, dateStr]
    for name in ['dataname', 'sbj', 'roi', 'method']:
        if hasattr(flags, name):
            parts.append(str(getattr(flags, name)).replace('/', '-'))
    if hasattr(flags, 'lambda_spatial') and flags.lambda_spatial > 0:
        parts.append('spatial_' + str(flags.lambda_spatial).replace('.', 'p'))
    if getattr(flags, 'run_tag', ''):
        parts.append(str(flags.run_tag).replace('/', '-').replace('\\', '-'))
    str_experiments = '_'.join(parts);
    return str_experiments

def create_dir_structure(flags, train=True):
    if train:
        str_experiments = get_str_experiments(flags)
        flags.dir_experiment_run = os.path.join(flags.dir_experiment, str_experiments)
        flags.str_experiment = str_experiments;
    else:
        flags.dir_experiment_run = flags.dir_experiment;

    print(flags.dir_experiment_run)
    if train:
        create_dir(flags.dir_experiment_run)

    flags.dir_checkpoints = os.path.join(flags.dir_experiment_run, 'checkpoints')
    if train:
        create_dir(flags.dir_checkpoints)

    flags.dir_logs = os.path.join(flags.dir_experiment_run, 'logs')
    if train:
        create_dir(flags.dir_logs)
    print(flags.dir_logs)
    return flags;
