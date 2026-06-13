# Notes: Reinforcement Learning Basics (Interview Prep)

Date: 2026-06-01
Tags: interview-prep, reinforcement-learning

## Core Loop

An RL agent interacts with an environment in a loop: observe state ->
choose action -> receive reward and next state -> update policy. The goal
is to learn a policy that maximizes cumulative reward over time, not just
immediate reward.

## Q-Learning vs SARSA

Both are temporal-difference learning methods for estimating the value of
state-action pairs (Q-values).

- **Q-Learning** is off-policy: it updates Q(s,a) using the *maximum*
  possible Q-value of the next state, regardless of what action the agent
  actually takes next. This means it learns the optimal policy even while
  exploring with a different (e.g., random) policy.

- **SARSA** is on-policy: it updates Q(s,a) using the Q-value of the
  action the agent *actually takes* next (following its current policy,
  including exploration). This makes SARSA more conservative — it learns
  a policy that accounts for the exploration strategy itself.

Practical difference: in a grid world with a "cliff" the agent can fall
off, Q-learning will learn the shortest path along the cliff edge (since
it assumes optimal future actions), while SARSA learns a safer path
further from the cliff (since it accounts for the chance of a random
exploratory step causing a fall).

## Why This Matters for RLHF

Modern LLM fine-tuning with RLHF (Reinforcement Learning from Human
Feedback) uses a reward model trained on human preference data, then
optimizes the policy (the LLM) against that reward model — typically with
PPO (Proximal Policy Optimization), a policy-gradient method rather than
Q-learning, because the action space (next-token distribution) is huge and
continuous-ish.

## DPO as an Alternative to RLHF/PPO

Direct Preference Optimization (DPO) skips training a separate reward
model entirely. Instead, given pairs of (preferred, rejected) responses,
DPO directly optimizes the policy to increase the relative likelihood of
preferred responses over rejected ones, using a loss derived from the
same Bradley-Terry preference model that would otherwise train a reward
model. Simpler pipeline, fewer moving parts, often comparable results to
PPO-based RLHF for many tasks.

## Constitutional AI (high level)

Anthropic's approach: instead of relying purely on human-labeled
preferences, use a set of written principles ("constitution") and have the
model critique and revise its own outputs against those principles,
generating training data for both supervised fine-tuning and the
preference-modeling stage. Reduces reliance on large volumes of human
labels for harmlessness training specifically.
