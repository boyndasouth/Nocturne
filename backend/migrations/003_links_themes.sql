-- Phase 3, step 2: Linker and Abstractor.

-- A theme note points at the facts it summarises: theme --summarizes--> fact. Kept apart
-- from the Linker's fact-to-fact types so retrieval can expand theme -> members without
-- confusing membership with "example_of" between two facts.
-- (ADD VALUE is allowed inside a transaction since PG 12; the value is first used later.)
alter type link_type add value if not exists 'summarizes';

-- Per-cycle counters for the new sub-agents (spec §4.4). Themes the Abstractor retires go
-- in the existing `archived` counter.
alter table sleep_cycles
    add column links_created  int not null default 0,
    add column themes_created int not null default 0,
    add column themes_updated int not null default 0;

-- Theme membership lookups ("which facts does this theme summarise?") filter by type.
create index memory_links_src_type on memory_links (src_id, link_type);
