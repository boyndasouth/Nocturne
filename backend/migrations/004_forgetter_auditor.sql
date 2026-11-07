-- Phase 3, step 4: Forgetter and Auditor.

-- Auditor (spec §3.2 step 7): LLM-as-judge quality of a sample of each cycle's decisions.
-- audit_score is the mean over sampled items (good = 1, questionable = 0.5, wrong = 0);
-- audit holds the per-item verdicts, or the error if the audit couldn't run. NULL = not
-- audited (auditor off, or nothing to audit).
alter table sleep_cycles
    add column audit_score real check (audit_score is null or (audit_score >= 0 and audit_score <= 1)),
    add column audit_items int not null default 0,
    add column audit jsonb;

-- The Forgetter scans active facts and consolidated episodes every cycle.
create index memory_notes_forget_scan on memory_notes (tier, status)
    where status = 'active' and tier in ('fact', 'episode');
