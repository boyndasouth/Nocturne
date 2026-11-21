-- A memory the person asked to forget. Unlike 'archived' (the Forgetter's soft forgetting,
-- which retrieval can fall back to), nothing ever retrieves or consolidates a forgotten
-- note. The row stays so its history is still traceable.
alter type note_status add value if not exists 'forgotten';
