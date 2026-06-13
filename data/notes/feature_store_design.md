# Notes: Feature Store Design

Date: 2026-04-12
Tags: system-design, ml-infra

## What is a Feature Store?

A feature store is a data system that sits between raw data sources and ML
models, providing a unified way to define, compute, store, and serve
features for both training and inference.

## Two Storage Layers

- **Offline store**: holds historical feature values, usually in a data
  warehouse or columnar format (Parquet, BigQuery). Used to generate
  training datasets.
- **Online store**: holds the latest feature values, optimized for
  low-latency point lookups (Redis, DynamoDB, Cassandra). Used at
  inference time.

## Point-in-Time Correctness

The hardest problem in feature stores. When generating training data, you
must join feature values "as they were" at the time of the label event —
not the current value. Getting this wrong causes **training/serving skew**
and label leakage, where the model sees information from the future during
training that it would never have at inference time.

Common approach: store feature values with timestamps, and use a
point-in-time join (ASOF join) when building training sets.

## Feature Freshness vs Cost Tradeoff

Streaming feature computation (Kafka + Flink) gives near-real-time
freshness but costs more in infra and complexity. Batch computation
(daily Airflow jobs) is cheaper but features can be stale by up to 24h.
Most systems use a hybrid: critical features streamed, everything else
batched.

## Open Questions

- How do you handle backfills when a feature definition changes?
- Versioning feature definitions so old models still get consistent
  features even after a schema change.
