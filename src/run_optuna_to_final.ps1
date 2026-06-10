# This assumes that the official MATLAB/XML-RPC server is already running on port 8081.
python src/run_optuna_to_final.py `
  --address 127.0.0.1 `
  --port 8081 `
  --optuna-budget 500 `
  --optuna-seeds 316018,348556,1001 `
  --optuna-trials 12 `
  --optuna-out results/optuna_tuning `
  --final-template cases/final_comparison_10seeds_simple.csv `
  --final-cases cases/final_comparison_10seeds_optuna.csv `
  --final-out results/final_10seeds_optuna
