# Debug Log: MatchEvent Search Returning Empty Results

Date: 2026-05-03
Tags: debugging, backend, matching-service

## Symptom

The match console frontend shows zero results when searching for
MatchEvent entities by GUID, even though the events clearly exist in
MongoDB (verified via direct query in mongosh).

## Investigation

1. Checked the frontend network tab — the API request to
   `/api/v1/matchEvents/search` returns 200 with an empty array, so this
   isn't a frontend rendering bug.
2. Added logging to the backend search handler. The MongoDB query being
   executed is:
   ```
   db.persistentEventMatchAttempt.find({ guid: "<the-guid>" })
   ```
3. Ran this query directly in mongosh — it takes 8-12 seconds and returns
   the correct document. In the application, the query times out after 5
   seconds (configured timeout) and the handler swallows the timeout
   error, returning an empty list instead of propagating the error.

## Root Cause Hypothesis

The `persistentEventMatchAttempt` collection has no index on the `guid`
field. With several million documents, a query on `guid` triggers a full
collection scan, which exceeds the 5-second application timeout under
load. The empty-result behavior is a secondary bug: timeouts should
surface as errors, not silently return "no results."

## Fix Plan

- Add an index: `db.persistentEventMatchAttempt.createIndex({ guid: 1 })`
- Separately, fix the error handling so a timeout returns a 504 with a
  clear message instead of an empty 200 — silent failures like this are
  why it took two days to notice.
- After adding the index, re-run the same query and confirm it returns in
  under 50ms.

## Lesson

Silent failure modes (catching an error and returning an empty/default
value) are dangerous because they look like "no data" instead of "system
broken." Prefer to propagate errors with clear status codes, and reserve
empty results for genuinely empty queries.
