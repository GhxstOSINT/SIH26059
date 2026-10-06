# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The primary user for the new decision view is a vessel operator monitoring a voyage. Government analysts and independent science, maritime, security, operations, and release reviewers use the supporting case workflow.

## Product Purpose

Southern Passage brings Antarctic ice observations, route context, model evidence, and human review into one workspace. It should help users see what the evidence says, what it does not say, and what changes a route assessment. Success is a reproducible, appropriately cautious evidence handoff—not automatic navigation clearance.

## Operating Context

The existing web portal and versioned API can run on a private host and be placed behind a ministry-approved OIDC gateway. The current route display includes illustrative scenarios and dated satellite observations. No live vessel-position feed or approved Polar Water Operational Manual profile is connected.

## Capabilities and Constraints

The system has dated observed sea-ice route sampling, research forecast diagnostics, historical iceberg context, a source-bound JSON review packet, and a role-gated human review ledger. The requested additions are route evidence/abstention, sensitivity to changed conditions, vessel-specific screening, and portable GIS evidence. Vessel limits are research inputs until an authoritative profile registry and maritime approval exist. Retrospective data and model results must not be presented as live navigation guidance.

## Brand Commitments

The existing product name is Southern Passage. The established visual system and compact operator workspace in the current portal are incumbent design authority; this feature extends them rather than replacing the identity.

## Evidence on Hand

The project contains checksummed NOAA VIIRS observations, Copernicus sea-ice/current/wind subsets, research model evaluations, historical iceberg records, and API tests. The latest matched Copernicus wind/current model did not beat persistence on the held-out August 2024 period. Actual vessel outcomes, live vessel telemetry, forecast-origin-verified forcing, and an approved vessel profile are not available.

## Product Principles

- Show provenance, coverage, and missing evidence at the point of use.
- Abstain when observations or validation do not support a decision.
- Make changed assumptions and their effect visible without calling a hypothetical scenario a forecast.
- Keep vessel safety decisions with authorized humans and approved policies.
- Export reproducible evidence for existing ministry systems.
