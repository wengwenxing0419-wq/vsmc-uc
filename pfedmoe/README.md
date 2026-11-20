## pFedMoE

This package adds a minimally invasive `pFedMoE` baseline to the existing
federated Switch Transformer training stack.

Enable it through the main entrypoint:

```bash
python train.py \
  --aggregation_method pfedmoe \
  --pfedmoe_enabled true
```

The shared/public training knobs still come from `args.py`, including
`--num_clients`, `--server_epochs`, `--client_epochs`, `--batch_size`,
`--learning_rate`, `--backbone_name`, `--data_save_path`, `--model_save_path`,
and `--save_result`.
