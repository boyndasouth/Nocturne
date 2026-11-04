-- Phase 2: sleep-cycle bookkeeping, full-row version snapshots, per-note decay τ.

-- §4.3: snapshot the whole row (minus the derived tsvector) so a cycle can be rolled back.
alter table memory_versions add column snapshot jsonb;
update memory_versions set snapshot = jsonb_build_object('content', content) where snapshot is null;
alter table memory_versions alter column snapshot set not null;
alter table memory_versions add column created_at timestamptz not null default now();
create index memory_versions_cycle on memory_versions (changed_by_cycle);

-- §5 forgetting curve: τ grows with each access (spacing effect), so it's per note.
-- Used by the Forgetter in Phase 3.
alter table memory_notes add column decay_tau_days real not null default 30.0
    check (decay_tau_days > 0);

-- Which sleep cycle consolidated an episode. NULL = pending. Bookkeeping, not a content
-- edit, so setting it doesn't bump `version` (same rule as access reinforcement).
alter table memory_notes add column consolidated_in uuid references sleep_cycles (cycle_id);
create index memory_notes_pending_episodes on memory_notes (recorded_at)
    where tier = 'episode' and consolidated_in is null;

-- §4.4 additions for checkpointed, resumable cycles.
alter table sleep_cycles
    add column episode_ids  uuid[] not null default '{}',  -- the batch this cycle owns
    add column clock        timestamptz not null default now(),  -- cycle time (simulated in replay)
    add column attempts     int not null default 1,
    add column heartbeat_at timestamptz not null default now(),
    add column llm_calls    int not null default 0,
    add column error        text;

-- At most one running cycle across all processes (idle trigger, cron, manual /sleep).
create unique index sleep_cycles_one_running on sleep_cycles ((true)) where status = 'running';
create index sleep_cycles_started on sleep_cycles (started_at desc);

-- Dedup/contradiction search active facts only. HNSW filters *after* the index scan, so once
-- episodes outnumber facts a shared index can return too few facts. A partial index matching
-- that predicate avoids it.
create index memory_notes_active_fact_hnsw on memory_notes
    using hnsw (embedding vector_cosine_ops) where tier = 'fact' and status = 'active';
