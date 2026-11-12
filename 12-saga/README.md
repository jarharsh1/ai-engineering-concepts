# 12 · Saga

**Level 4 · Architecture · no API key**

## What it is
A Saga manages a transaction that spans multiple services where a single ACID
transaction is impossible. It runs a sequence of **local** steps, and each step
has a **compensating action** that semantically undoes it. If a step fails, the
orchestrator runs the compensations for the already-completed steps **in reverse
order**, restoring consistency without a global lock.

## Why it matters in production
Microservices and agentic tool-chains constantly do multi-step work — reserve
inventory, charge a card, confirm, notify — across systems that don't share a
database. Without sagas, a mid-way failure leaves money charged for an order that
never shipped. Sagas give you a disciplined rollback story (orchestrated or
choreographed) for exactly these partial failures.

## Where it's used
Order/booking/payment workflows, microservice orchestration, event-driven
systems, and multi-tool agent plans that must undo side effects on failure.

## Run it
```bash
python 12-saga/demo.py
```
Runs a Saga orchestrator with forward steps (search→reserve→charge→confirm→update),
each with a compensating action. Shows the happy path, then a failure at step 4
that runs compensations in reverse order.
