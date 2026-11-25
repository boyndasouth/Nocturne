-- Daily usage, so a public deployment can't burn the free LLM quotas.
--
-- usage_daily: what each account did today (messages, sleeps it asked for). Private to the
-- account, like its memory.
-- llm_usage_daily: what each model budget spent today across everyone ('answer' is the chat
-- model, 'sleep' the consolidation model, which also answers when 'answer' is spent).
-- Shared counters with no personal data, so no row-level security.

create table usage_daily (
    owner_id text not null
        default coalesce(nullif(current_setting('nocturne.owner', true), ''), 'local'),
    day date not null,
    chats integer not null default 0,
    sleeps integer not null default 0,
    primary key (owner_id, day)
);

create table llm_usage_daily (
    day date not null,
    budget text not null,
    requests integer not null default 0,
    tokens bigint not null default 0,
    primary key (day, budget)
);

grant select, insert, update on usage_daily, llm_usage_daily to nocturne_app;

alter table usage_daily enable row level security;
create policy own_rows on usage_daily to nocturne_app
    using (owner_id = current_setting('nocturne.owner', true))
    with check (owner_id = current_setting('nocturne.owner', true));
