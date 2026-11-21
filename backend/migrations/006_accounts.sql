-- Accounts: every memory row belongs to an owner, and Postgres enforces it.
--
-- With AUTH_MODE=supabase the app switches each pooled connection to the restricted role
-- nocturne_app and sets nocturne.owner to the signed-in user's id; the policies below then
-- hide every other owner's rows, from chat, the explorer and the sleep agent alike, and
-- reject writes for anyone else. A connection with no owner set sees nothing.
--
-- With AUTH_MODE=none (local dev, the eval harness, tests) connections stay the table
-- owner, which row-level security doesn't apply to, and new rows default to 'local'.
-- Queries behave exactly as before.

do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'nocturne_app') then
        create role nocturne_app nologin;
    end if;
end
$$;
-- The app's own login role must be allowed to switch to it.
grant nocturne_app to current_user;

grant usage on schema public to nocturne_app;
grant usage on type note_tier, note_status, link_type, sleep_status to nocturne_app;
grant select, insert, update on memory_notes, memory_links, memory_versions, sleep_cycles
    to nocturne_app;
grant delete on memory_links to nocturne_app;  -- theme membership is replaced on update

alter table memory_notes add column owner_id text not null default 'local';
alter table memory_links add column owner_id text not null default 'local';
alter table memory_versions add column owner_id text not null default 'local';
alter table sleep_cycles add column owner_id text not null default 'local';

-- Existing rows (above) belong to 'local'; new ones to whoever the connection is acting for.
alter table memory_notes alter column owner_id
    set default coalesce(nullif(current_setting('nocturne.owner', true), ''), 'local');
alter table memory_links alter column owner_id
    set default coalesce(nullif(current_setting('nocturne.owner', true), ''), 'local');
alter table memory_versions alter column owner_id
    set default coalesce(nullif(current_setting('nocturne.owner', true), ''), 'local');
alter table sleep_cycles alter column owner_id
    set default coalesce(nullif(current_setting('nocturne.owner', true), ''), 'local');

create index memory_notes_owner on memory_notes (owner_id, tier, status);
create index sleep_cycles_owner on sleep_cycles (owner_id, started_at desc);

-- One running cycle per owner instead of one overall.
drop index sleep_cycles_one_running;
create unique index sleep_cycles_one_running on sleep_cycles (owner_id) where status = 'running';

alter table memory_notes enable row level security;
alter table memory_links enable row level security;
alter table memory_versions enable row level security;
alter table sleep_cycles enable row level security;

create policy own_rows on memory_notes to nocturne_app
    using (owner_id = current_setting('nocturne.owner', true))
    with check (owner_id = current_setting('nocturne.owner', true));
create policy own_rows on memory_links to nocturne_app
    using (owner_id = current_setting('nocturne.owner', true))
    with check (owner_id = current_setting('nocturne.owner', true));
create policy own_rows on memory_versions to nocturne_app
    using (owner_id = current_setting('nocturne.owner', true))
    with check (owner_id = current_setting('nocturne.owner', true));
create policy own_rows on sleep_cycles to nocturne_app
    using (owner_id = current_setting('nocturne.owner', true))
    with check (owner_id = current_setting('nocturne.owner', true));
