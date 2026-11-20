This folder implements a FedMix-style mixed-supervision baseline for FL + Switch Transformer / MoE.

The baseline is isolated from the original training code and reuses the shared experiment arguments from `args.py`
so comparisons keep the same public settings:

- dataset and data split paths
- number of clients
- local epochs
- optimizer and learning-rate configuration
- random seed
- model architecture
- evaluation pipeline

FedMix-specific behavior in this adaptation:

- local training keeps the original supervised objective and adds teacher-student pseudo supervision
- dynamic sample selection keeps only confident agreement samples when enabled
- non-expert parameters stay on the shared FedAvg path
- expert parameters use FedMix-style adaptive aggregation weights based on sample count and client loss

Run example:

```bash
cd /home/wwx/Vsmc-2
source /home/wwx/code/vsmc/bin/activate

python -m baselines.fedmix_mixed_supervision.run_fedmix \
  --data_name cifar10 \
  --num_clients 10 \
  --alpha 0.1 \
  --seed 3 \
  --device cuda \
  --server_epochs 50 \
  --client_epochs 1 \
  --backbone_name resnet18 \
  --fedavg_weight_mode sample_count \
  --expert_weight_mode uniform \
  --fedmix_beta 1.5 \
  --fedmix_lambda 10.0 \
  --fedmix_pseudo_thresh 0.9 \
  --data_save_path experiments/run_1/data_alpha0.1_10client_seed3 \
  --output_dir experiments/run_2/fedmix_alpha0.1/seed3
```

Artifacts are written inside the FedMix run directory:

- `logs/fedmix.log`
- `logs/round_metrics.csv`
- `logs/run_summary.txt`
- `models/server.pth`
- `detail/data_*.csv`
