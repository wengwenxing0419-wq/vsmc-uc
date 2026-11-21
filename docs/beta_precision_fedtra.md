# Dynamic Beta-Precision FedTRA

This branch adds a separate beta_precision_fedtra aggregation method. It does
not replace or modify the existing vsmc_expert baseline.

## Method

- u: training-time expert routing fraction from client metrics.
- c: mapped leave-one-out cosine agreement between each active client's
  expert update and the sum of the other active clients' normalized updates.
- Each client-expert pair maintains a scalar latent reliability state and a
  bootstrap particle-filter approximation to its filtering posterior.
- The two observation channels use conditionally independent Beta likelihoods.
  Their concentrations are K0 * gamma_u and K0 * gamma_c, where
  K0 = kappa_u + kappa_c is fixed.
- Initial shares are kappa_u/K0 and kappa_c/K0, recovering the configured
  base Beta concentrations at initialization.
- The OMD feedback is the gradient of the Jensen upper-bound surrogate of the
  one-step predictive negative log likelihood. The exact predictive NLL and
  surrogate NLL are both logged.

## Run

Use the existing training command and select:

    --aggregation_method beta_precision_fedtra

Optional parameters:

    --beta_precision_num_particles 32
    --beta_precision_kappa_u 10
    --beta_precision_kappa_c 10
    --beta_precision_omd_lr 0.05
    --beta_precision_gamma_min 0.001
    --beta_precision_transition_rho 0.9
    --beta_precision_transition_std 0.15

Logs are written under the experiment's logs/ directory:

- beta_precision_fedtra_round_metrics.csv
- beta_precision_fedtra_pair_metrics.csv

## Important implementation notes

1. Client training routing statistics are sent separately from validation
   routing statistics. The u observation uses training-time routing counts.
2. c is calculated from the client expert parameter deltas relative to the
   previous global expert. Inactive clients are excluded from the leave-one-out
   consensus and expert aggregation.
3. The latent-state filter in this implementation is a bootstrap particle
   filter. It is not the learned residual-Gaussian proposal VSMC inference
   procedure described in the original FedTRA paper.
4. The convexity and OMD regret argument applies to the Jensen surrogate with
   fixed predictive particles and bounded gradients. It is not a guarantee of
   improved test accuracy.
5. This is an experimental implementation. Run unit tests and controlled
   baseline comparisons before treating it as a validated research result.
