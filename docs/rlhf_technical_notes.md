# RLHF Technical Notes

This note explains the mechanics behind the RLHF extension. It is meant as the deeper companion to `docs/rlhf_readme.md`.

## 1. SFT objective

The supervised fine-tuning stage trains the policy to imitate the preferred HelpSteer3 response. Given a rendered prompt `x` and preferred response `y`, we train the causal LM on the concatenated sequence:

```text
[prompt tokens][assistant response tokens]
```

The loss is masked on the prompt tokens and computed only on assistant response tokens:

```text
L_SFT(theta) = - sum_t log pi_theta(y_t | x, y_<t)
```

In the final run, SFT used a 4096-token total sequence limit. That matters because HelpSteer3 often contains long prompts and long responses. At 1024 tokens, about 38% of SFT chosen responses were truncated; at 4096, less than 1% were truncated.

## 2. Reward model

The reward model takes a complete prompt-response pair and returns a scalar score:

```text
r_phi(prompt, response) -> real number
```

It is trained from chosen/rejected HelpSteer3 pairs using a Bradley-Terry / logistic ranking loss:

```text
L_RM(phi) = - log sigmoid(r_phi(chosen) - r_phi(rejected))
```

A good reward model should assign higher score to the chosen response than the rejected response. The final 4096-token epoch-2 reward model reached 71.62% validation accuracy on 1917 preference pairs.

### Interpreting negative rewards

The reward model's scalar output is not calibrated to an external human score. Adding a constant to all rewards would not change pairwise preferences. The sign is therefore not intrinsically meaningful. A response with score `-3.5` can still be better than another response with score `-5.0` for the same prompt.

The final reward histograms look roughly bell-shaped because most prompt-response pairs fall into the model's normal scoring range, while unusually preferred or dispreferred examples form tails. This is not automatically a bug. The key quantities are margins and pairwise rankings.

## 3. PPO for language models

In LLM PPO, an action is a generated token. A rollout is a sampled assistant response. For a prompt `x`, the current policy samples:

```text
y ~ pi_theta(. | x)
```

The response is scored by the reward model, and a KL penalty discourages the policy from drifting too far from the frozen SFT reference policy:

```text
R_total = R_model - beta * KL(pi_theta || pi_ref)
```

The PPO update uses response-token log-probabilities only. Prompt tokens are context; they are not actions and do not receive policy-gradient loss.

## 4. PPO clipped objective

For each generated token, PPO compares the new policy probability against the old rollout policy probability:

```text
ratio_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
```

The clipped PPO objective is:

```text
L_clip(theta) = E[min(ratio_t A_t, clip(ratio_t, 1-eps, 1+eps) A_t)]
```

The clipping prevents optimization from taking too much reward-model incentive from the same sampled token batch. This is the language-model analogue of the original PPO trust-region idea.

## 5. Why KL anchoring matters

The frozen reference model is the SFT model. PPO is allowed to improve against the reward model, but it is not allowed to drift arbitrarily. Without this anchor, reward hacking can produce degenerate text: empty answers, repetition, multilingual drift, or high-reward nonsense.

The final PPO run used strong KL settings:

```yaml
kl.init_kl_coef: 0.18
kl.min_kl_coef: 0.14
kl.max_kl_coef: 3.0
ppo.clip_range: 0.06
ppo.learning_rate: 3.0e-7
```

These settings are conservative enough to avoid collapse, but may also explain why PPO did not dramatically outperform SFT or base.

## 6. Token budgets

Qwen2.5-0.5B-Instruct supports much larger contexts than the final training run used. However, RLHF training is more expensive than inference because we must keep policy, reference, reward model, value head, log-probabilities, masks, and rollout tensors around.

Final budgets:

| Stage | Budget |
|---|---:|
| SFT | 4096 total tokens |
| Reward model | 4096 total tokens |
| PPO prompt | 3072 prompt tokens |
| PPO response | 512 new tokens |
| Evaluation prompt | 3072 prompt tokens |
| Evaluation response | 512 new tokens |

This is a compromise between data coverage and compute. It is much less constrained than the early 128-token generation runs, and it is large enough for qualitative examples.

## 7. Why PPO did not dominate

Several reasons are plausible:

1. **Base model is already instruction-tuned.** Qwen2.5-0.5B-Instruct is not an unaligned base LM. It is already post-trained.
2. **Reward model is imperfect.** 71.62% validation accuracy is useful but not equivalent to a human judge.
3. **Small model capacity.** A 0.5B policy has limited ability to improve while preserving broad capability.
4. **Conservative PPO.** Strong KL and low learning rate prevent collapse but limit improvement.
5. **Reward-model mismatch.** The reward model is trained on chosen/rejected pairs; PPO optimizes generated responses that may differ from the training distribution.
6. **Long generation variance.** 512-token generations create more opportunity for both helpful detail and reward-hacking/noisy continuations.

These constraints help explain why the small-model RLHF experiment produced stable training without a clear aggregate improvement.

## 8. Main artifacts

| Artifact | Purpose |
|---|---|
| `configs/rlhf/qwen25_05b_helpsteer3_sft.yaml` | final SFT config |
| `configs/rlhf/qwen25_05b_helpsteer3_reward.yaml` | final reward-model config |
| `configs/rlhf/qwen25_05b_helpsteer3_ppo.yaml` | final PPO config |
| `configs/rlhf/qwen25_05b_helpsteer3_eval_suite.yaml` | final suite-eval config |
| `outputs/rlhf/length_diagnostics/` | token truncation study |
| `outputs/rlhf/qwen25_05b_helpsteer3_sft_4096/` | SFT checkpoint and metrics |
| `outputs/rlhf/qwen25_05b_helpsteer3_reward_4096_epoch2/` | final reward model |
| `outputs/rlhf/qwen25_05b_helpsteer3_ppo_4096_epoch2_long512/` | final PPO checkpoint |
| `outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400/` | final Base/SFT/PPO eval |
