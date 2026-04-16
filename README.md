# trpo

There are tons of TRPO implementations out there already. This is my attempt at it.

The main command to perform training is to run this in the repository's root:

```src
python3 scripts/train.py --config configs/mujoco/cartpole_linear.yaml
```

The environment to train on should be specified using a config file.