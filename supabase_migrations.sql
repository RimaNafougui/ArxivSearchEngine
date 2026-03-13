-- ArXiv RAG — Supabase migrations
-- Run each statement in the Supabase SQL Editor (supabase.com → SQL Editor).
-- Safe to re-run: all statements use IF NOT EXISTS / DO NOTHING guards.

-- ── 1. Answer Quality Feedback ────────────────────────────────────────────────
-- Stores thumbs-up / thumbs-down ratings from the UI.
CREATE TABLE IF NOT EXISTS feedback (
    id        BIGSERIAL    PRIMARY KEY,
    query     TEXT         NOT NULL,
    answer    TEXT,
    rating    SMALLINT     NOT NULL CHECK (rating IN (-1, 1)),
    timestamp TIMESTAMPTZ  DEFAULT now()
);

-- ── 2. Paper Alert Subscriptions ──────────────────────────────────────────────
-- One row per subscriber; topics is a text array (e.g. '{diffusion models,LoRA}').
-- The UNIQUE constraint on email allows upsert (new topics replace old ones).
CREATE TABLE IF NOT EXISTS paper_alerts (
    id         BIGSERIAL    PRIMARY KEY,
    email      TEXT         NOT NULL,
    topics     TEXT[]       NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ  DEFAULT now(),
    UNIQUE (email)
);
