## FedJETs

This package adds a minimally invasive FedJETs comparison baseline on top of the
existing federated Switch Transformer training stack.

Enable it from the main training entrypoint with:

```bash
python train.py \
  --aggregation_method fedjets \
  --fedjets_enabled true
```

Common experiment settings such as `--num_clients`, `--server_epochs`,
`--client_epochs`, `--batch_size`, `--learning_rate`, `--backbone_name`,
`--data_save_path`, `--model_save_path`, and `--save_result` are still inherited
from `args.py`.
