# RLHF Run Inspection
## Reward model
- Validation pairwise accuracy: **0.6646**
- Validation loss: **1.2450**
- Average reward margin: **5.0151**

## PPO training
- Updates: **250**
- reward_model_score: first=-1.1289, last=-2.5711, min=-5.5546, max=3.1909, mean=-0.9823
- objective_kl: first=-0.0018, last=-0.0366, min=-0.0373, max=0.0042, mean=-0.0111
- abs_ref_logratio: first=0.0302, last=0.1497, min=0.0269, max=0.1624, mean=0.0735
- kl_coef: first=0.1000, last=0.1000, min=0.1000, max=0.1000, mean=0.1000
- clip_fraction: first=0.0921, last=0.1183, min=0.0724, max=0.1476, mean=0.1099
- mean_response_tokens: first=85.4375, last=81.3750, min=78.0000, max=96.0000, mean=89.8077

## Before/after evaluation
- Examples: **200**
- Winner counts: `{'base': 113, 'ppo': 87}`
- Mean reward delta: **-0.2813**
- PPO responses with bad/debug patterns: **1**
- PPO responses with CJK characters: **12**
  - ⚠️ Base model wins more examples than PPO under the reward model.
  - ⚠️ PPO responses contain known degenerate/toxic/debug-token patterns.
  - ⚠️ PPO responses contain CJK characters; verify multilingual prompts or drift.
