# Notes: PromQL Patterns for Dashboards

Date: 2026-04-28
Tags: observability, grafana, promql

## Summing Across Multiple Series with `or`

When a metric is emitted by multiple sources but you want one combined
total line on a dashboard, a naive `sum(metric_name)` can produce
unexpected results if some label combinations don't exist for all time
ranges (Grafana shows gaps).

Pattern that works well — use `or` inside the `sum()` to provide a
zero-fallback series:

```promql
sum(
  rate(matcher_count{type="event"}[5m])
  or
  vector(0)
)
```

This guarantees the query always returns a value (0 if no data) instead
of "no data" gaps on the graph.

## Trailing Pipe Bug in Variable Filters

Grafana template variables that build a regex filter (e.g., a multi-select
dropdown of "type" values) can produce a query like:

```promql
matcher_count{type=~"event|person|"}
```

Notice the trailing `|` — this happens when the variable's "All" option is
included alongside specific selections, and the join logic adds an empty
string. The trailing `|` makes the regex match an empty string too, which
can silently include unintended series.

Fix: filter out empty strings before joining the variable values, or use
`type=~"^(event|person)$"` with explicit anchors.

## Debugging Missing Metric Values

When `title_retry` was missing from `matcher_count` entirely (not zero —
absent), the cause was that the counter is only incremented inside a
specific retry code path, and that code path had never been exercised in
the time window being viewed. Prometheus counters that have never been
incremented simply don't exist as a series — they don't show up as 0,
they don't show up at all. This is different from a metric that exists but
is currently 0.

Takeaway: "missing from the dashboard" and "value is zero" are different
failure modes with different causes — always check whether the series
exists at all (`count(metric_name)`) before assuming the value is 0.
