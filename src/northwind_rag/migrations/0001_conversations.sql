-- Conversations and their messages. Content is stored AFTER PII redaction: the raw user input
-- never reaches the database, exactly like it never reaches the LLM or the support desk.

CREATE TABLE conversations (
    id              text        PRIMARY KEY CHECK (length(id) BETWEEN 1 AND 128),
    user_id         text        NOT NULL,                 -- durable owner (Redis only caches it)
    status          text        NOT NULL DEFAULT 'bot'
                                CHECK (status IN ('bot', 'agent', 'closed')),
    title           text,                                 -- first user message, truncated
    handoff_ticket  text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- "my conversations, newest first" with keyset pagination
CREATE INDEX conversations_user_recent ON conversations (user_id, updated_at DESC, id DESC);
-- retention purge
CREATE INDEX conversations_updated ON conversations (updated_at);

CREATE TABLE messages (
    id               bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id  text        NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role             text        NOT NULL CHECK (role IN ('user', 'assistant', 'agent', 'system')),
    content          text        NOT NULL,
    status           text,                                -- assistant: answered / no_context / blocked ...
    metadata         jsonb       NOT NULL DEFAULT '{}',   -- citations, grounding, handoff, agent name ...
    idempotency_key  text,                                -- client message id / desk event id
    created_at       timestamptz NOT NULL DEFAULT now(),
    -- retried requests and redelivered webhooks are stored once (NULL keys are never equal)
    CONSTRAINT messages_idempotent UNIQUE (conversation_id, idempotency_key)
);

CREATE INDEX messages_by_conversation ON messages (conversation_id, id);
