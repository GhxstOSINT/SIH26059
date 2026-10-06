# Southern Passage standalone research service: operator runbook

Southern Passage can run independently on a private host and later be placed behind a ministry-approved portal gateway. The current service is a research observation and review system, **not a navigation decision service**. A successful deployment or smoke check does not establish scientific skill, source freshness, SSO assurance, or maritime safety.

## Local or private staging

1. Use the pinned Python environment or build the provided container. The Docker Compose example binds the API and portal to `127.0.0.1:8000` only; it is not a public deployment definition. The Docker engine was unavailable in the development environment when this runbook was written, so the image build still requires verification on an operator-controlled host.
2. Verify the selected model and frozen evidence: `work\convlstm-env\Scripts\python.exe -m ml.verify_initial_model`.
3. Start the service locally: `work\convlstm-env\Scripts\python.exe -m uvicorn service.main:app --host 127.0.0.1 --port 8000`, or run `docker compose up --build` after reviewing mounted paths and settings.
4. Run `work\convlstm-env\Scripts\python.exe -m ops.smoke --base-url http://127.0.0.1:8000`. The script reads `SOUTHERN_PASSAGE_API_TOKEN` from the environment if the private API requires it; it does not print the token. A zero exit means the *research service package* works. Inspect `active_observation_status` and `review_capabilities.enabled` separately.
5. Do not expose the local development mode to remote users. For a hosted service, use `APP_ENV=production`, a secret-manager-provided 32+ character workload bearer, exact HTTPS CORS origins, private ingress, and a vetted OIDC gateway protecting both `/portal/` and `/api/v1/`. The browser must never hold the workload bearer. The gateway maps authenticated groups to application roles and signs request-bound identity assertions as specified in [MINISTRY_INTEGRATION.md](MINISTRY_INTEGRATION.md). This product does not invent its own ministry identities.

## Observation maintenance

The API reads mounted observations and never promotes a new source on its own. `ml/refresh_viirs.py` is a separate checksum-verified ingestion worker. Its checked-in policy is only a research example. A named data owner must approve product priority, age limits, minimum coverage, and alert contacts before scheduling it. Run one refresh invocation under an exclusive scheduler lock; alert on nonzero exit, inspect the retained rejection report, and do not silently treat a fallback sensor as equivalent. The source monitor and active feed status are visible through `/api/v1/integration/status` and `/api/v1/observations/active/status`. A stale or missing feed must remain unavailable, not green.

The frozen cross-platform corridor report is in [outputs/viirs-2024-multiplatform-frozen-corridors.json](outputs/viirs-2024-multiplatform-frozen-corridors.json). Its `insufficient_evidence` gate is expected. Do not reroute the fixed transects after seeing their scores or interpret a small score difference on sparse points as operational skill. The next scientific release requires new predeclared routes/seasons, denser independent observations, actual vessel outcomes, calibrated uncertainty, and science/maritime review.

## Review-ledger backup and recovery

If the SSO review workflow is enabled, back up its SQLite store from the host or a controlled maintenance job using the online backup API, not a live file copy:

```powershell
work\convlstm-env\Scripts\python.exe -m ops.review_backup --source work/review/reviews.sqlite3 --output work/backups/review-20261002.sqlite3
work\convlstm-env\Scripts\python.exe -m ops.review_backup --verify-only --output work/backups/review-20261002.sqlite3
```

Use a unique output name per run. The tool refuses overwrite, checks SQLite integrity and foreign keys, verifies every saved packet/event hash chain, and emits a SHA-256 manifest. Its manifest is **not** an immutable or signed audit record: an administrator who can rewrite the database can also rewrite the manifest. Copy verified backups to encrypted, access-controlled storage, test restoration into a separate path, and replicate event or chain-head evidence to an approved append-only audit destination. Never restore over a running database; stop the service, preserve the current file, restore a verified backup to a new path, and switch the configured path only after a rehearsal and authorization.

## Daily operator checks and incident rule

- Confirm process health and `/readyz`, then run `ops.smoke`; alert on nonzero exit. `/readyz` covers the pinned research model, not live-data availability.
- Inspect upstream source timestamps, active-scene age/QA, disk capacity, review-ledger backup age, and evidence-report integrity. Alert on failed refreshes and stale scenes.
- Check gateway sign-in/role denials and actor-attributed audit delivery. Disabled review workflow is a valid local research state, not a successful SSO deployment.
- If any source, model, review-ledger integrity, or identity control fails, keep research results unavailable or clearly degraded. Do not permit automatic route clearance.

## Remaining external acceptance

The ministry or an authorized deployment partner must supply and approve identity-provider configuration, role assignments, network and TLS boundary, secret manager, external evidence/document repository, immutable audit destination, backup retention, alert ownership, vessel constraints, route-outcome records, and maritime/scientific acceptance criteria. Without these, the independently hosted product may support controlled research review but cannot truthfully be called a ministry-certified operational system.
