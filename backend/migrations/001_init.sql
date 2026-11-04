-- Nocturne schema v1 (spec §4).
-- Works on Supabase, Neon, and the pgvector/pgvector Docker image.

create extension if not exists vector;

create type note_tier as enum ('episode', 'fact', 'theme');
create type note_status as enum ('active', 'superseded', 'archived');
create type link_type as enum ('related', 'elaborates', 'example_of', 'caused_by', 'supersedes');
create type sleep_status as enum ('running', 'succeeded', 'failed');

-- §4.4 ------------------------------------------------------------------------------------
create table sleep_cycles (
    cycle_id           uuid primary key default gen_random_uuid(),
    started_at         timestamptz not null default now(),
    finished_at        timestamptz,
    episodes_processed int not null default 0,
    facts_created      int not null default 0,
    merges             int not null default 0,
    supersessions      int not null default 0,
    archived           int not null default 0,
    tokens_used        int not null default 0,
    status             sleep_status not null default 'running'
);

-- §4.1 ------------------------------------------------------------------------------------
create table memory_notes (
    id                 uuid primary key default gen_random_uuid(),
    tier               note_tier not null,
    content            text not null,
    keywords           text[] not null default '{}',
    embedding          vector(384),
    strength           real not null default 1.0 check (strength >= 0 and strength <= 1),
    access_count       int not null default 0,
    last_accessed      timestamptz,
    valid_from         timestamptz not null default now(),  -- valid time
    valid_to           timestamptz,
    recorded_at        timestamptz not null default now(),  -- transaction time
    status             note_status not null default 'active',
    source_episode_ids uuid[] not null default '{}',
    version            int not null default 1,
    -- Derived full-text index column, maintained by trigger below (not part of the logical model).
    search_tsv         tsvector,
    check (valid_to is null or valid_to >= valid_from)
);

create index memory_notes_embedding_hnsw on memory_notes
    using hnsw (embedding vector_cosine_ops);
create index memory_notes_search_tsv on memory_notes using gin (search_tsv);
create index memory_notes_tier_status on memory_notes (tier, status);
create index memory_notes_valid_time on memory_notes (valid_from, valid_to);

-- Content is weighted above keywords. array_to_string isn't IMMUTABLE, so this can't be a
-- generated column.
create function memory_notes_tsv_update() returns trigger
language plpgsql as $$
begin
    new.search_tsv :=
        setweight(to_tsvector('english', coalesce(new.content, '')), 'A') ||
        setweight(to_tsvector('english', array_to_string(new.keywords, ' ')), 'B');
    return new;
end
$$;

create trigger memory_notes_tsv
    before insert or update of content, keywords on memory_notes
    for each row execute function memory_notes_tsv_update();

-- Design rule (spec §3.2): archive or supersede, never hard-delete.
-- An operator can bypass for maintenance with: set local nocturne.allow_delete = 'on';
create function memory_notes_block_delete() returns trigger
language plpgsql as $$
begin
    if coalesce(current_setting('nocturne.allow_delete', true), 'off') <> 'on' then
        raise exception 'memory_notes rows are never hard-deleted; archive or supersede instead';
    end if;
    return old;
end
$$;

create trigger memory_notes_no_delete
    before delete on memory_notes
    for each row execute function memory_notes_block_delete();

-- §4.2 ------------------------------------------------------------------------------------
create table memory_links (
    src_id           uuid not null references memory_notes (id),
    dst_id           uuid not null references memory_notes (id),
    link_type        link_type not null,
    weight           real not null default 1.0,
    created_by_cycle uuid references sleep_cycles (cycle_id),
    primary key (src_id, dst_id, link_type),
    check (src_id <> dst_id)
);

create index memory_links_dst on memory_links (dst_id);

-- §4.3 ------------------------------------------------------------------------------------
create table memory_versions (
    note_id          uuid not null references memory_notes (id),
    version          int not null,
    content          text not null,
    changed_by_cycle uuid references sleep_cycles (cycle_id),
    reason           text,
    primary key (note_id, version)
);
