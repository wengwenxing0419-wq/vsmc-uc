This folder implements a FedMA-style matched averaging baseline for FL + Switch Transformer / MoE.

Original FedMA performs layer-wise matched averaging for CNN/LSTM by matching hidden elements before averaging.
This adapted baseline applies the same principle to MoE experts:
experts are matched across clients by similarity before aggregation.

This implementation does not modify the original Learned VSMC algorithm.
Non-expert parameters are aggregated by FedAvg.
Expert parameters are aggregated by matched averaging.

The runner intentionally reuses the shared experiment arguments from `args.py` so common settings stay aligned during baseline comparisons:

- model architecture
- number of clients
- local epochs
- optimizer and learning-rate settings
- dataset split paths
- total communication rounds
- random seed
- evaluation function

FedMA adds only a small set of baseline-specific arguments:

```bash
python -m baselines.fedma_matched_averaging.run_fedma \
  --data_name cifar10 \
  --num_clients 50 \
  --server_epochs 5 \
  --client_epochs 1 \
  --backbone_name resnet18 \
  --fedma_scope experts_only \
  --expert_signature weight_mean \
  --matching_method hungarian \
  --similarity cosine \
  --seed 1 \
  --output_dir experiments/baselines_fedma/fedma_experts_only_weight_mean_seed1
```

Outputs are saved into the FedMA run directory and kept separate from the original Learned VSMC logs:

- `logs/round_metrics.csv`
- `logs/matching_round_<round>.json`
- `logs/similarity_round_<round>.pt`
- `logs/run_summary.txt`
- `models/server.pth`
