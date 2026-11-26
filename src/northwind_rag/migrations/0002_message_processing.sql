-- Save-first turns: the user's (redacted) question is stored BEFORE the pipeline runs, with
-- state 'processing'. The existing UNIQUE (conversation_id, idempotency_key) makes that insert
-- the lock: a retried message_id hits the constraint instead of starting a second generation.
--
-- state         NULL = not a tracked turn (assistant/agent/system rows, rows from before 0002)
-- attempts      fencing token: bumped on every re-claim, so a pod that lost the row can't finish it
-- owner         worker (host:pid:nonce) generating the answer; its shutdown hook releases its rows
-- heartbeat_at  renewed while generating; the sweeper fails rows whose heartbeat stopped (pod died)

ALTER TABLE messages
    ADD COLUMN state        text CHECK (state IN ('processing', 'done', 'failed')),
    ADD COLUMN attempts     integer NOT NULL DEFAULT 1 CHECK (attempts >= 1),
    ADD COLUMN owner        text,
    ADD COLUMN heartbeat_at timestamptz,
    ADD COLUMN error        text;

-- The sweeper only ever looks at in-flight rows: a tiny partial index, whatever the table size.
CREATE INDEX messages_processing ON messages (heartbeat_at) WHERE state = 'processing';

-- Questions whose stream was aborted before 0002 were stored with {"aborted": true} and no
-- answer. Mark them failed so a retry with the same message_id is generated, as it was before.
UPDATE messages SET state = 'failed', error = 'aborted (before 0002)'
 WHERE role = 'user' AND metadata ->> 'aborted' = 'true';
