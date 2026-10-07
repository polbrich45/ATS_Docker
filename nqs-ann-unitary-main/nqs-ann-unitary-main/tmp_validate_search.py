from scripts.hyperparameter_grid import build_cli_args

args = build_cli_args('j1j2', {'use_sr': True, 'n_rot': 10, 'n_subspace': 8})
assert '--no-sr' not in args
assert '--n-rot' in args and '10' in args
assert '--n-subspace' in args and '8' in args

args = build_cli_args('j1j2', {'use_sr': False, 'n_rot': 12})
assert '--no-sr' in args and '12' in args
print('cli-assertions: ok')
