srun --nodes=1  -w taurusml7 --tasks-per-node=1 --cpus-per-task=4 --gres=gpu:1 --comment=no_monitoring -t 01:00:00 -p ml --pty bash

sinfo -p ml