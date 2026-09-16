-- Add expression index so duplicate-check queries on metadata->>'url' don't
-- cause full table scans (which time out as the documents table grows).
CREATE INDEX IF NOT EXISTS idx_documents_metadata_url
    ON documents ((metadata->>'url'));

-- ═══════════════════════════════════════════════════════════════════════════
-- Row-Level Security
-- ═══════════════════════════════════════════════════════════════════════════
-- NOTE: app.py and send_alerts.py currently authenticate with the Supabase
-- *service-role* key, which bypasses RLS entirely. So enabling RLS here does
-- NOT change the app's own behaviour at all — it only closes the hole where
-- the public/anon key (if ever used, e.g. from a future JS client, or simply
-- discovered from the project URL) would otherwise have unrestricted
-- read/write on every table. Zero-risk to apply as-is.
--
-- Policies below are scoped to exactly the operations app.py / send_alerts.py
-- perform (see grep results this migration was derived from — retrieve via
-- hybrid_search/match_documents RPCs + table() calls). Review before running:
-- I could not verify this against the live project (a different, unrelated
-- Supabase project was connected when this was written), so double-check
-- column/table names match your actual schema first.
--
-- If you later want real defense-in-depth (not just "harmless because the
-- app doesn't use this key"), switch app.py/send_alerts.py off the
-- service-role key onto a scoped key that these policies actually govern —
-- that's a separate follow-up, test it against a staging query first.

-- ── documents: public, read-only data (paper chunks). Safe to expose broadly.
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "documents_select_all" ON documents;
CREATE POLICY "documents_select_all"
    ON documents FOR SELECT
    TO anon, authenticated
    USING (true);
-- No INSERT/UPDATE/DELETE policy for anon/authenticated — only the ETL
-- pipeline (service-role key, bypasses RLS) writes to this table.

-- ── feedback: ratings + query/answer text. Not PII, but also not meant to be
-- bulk-readable; SELECT is only granted because the app's hero-stats counter
-- does a public COUNT(*) against it. If you'd rather not expose even that,
-- replace this SELECT policy with a SECURITY DEFINER function that returns
-- just the count, and drop public SELECT entirely.
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "feedback_insert_anon" ON feedback;
CREATE POLICY "feedback_insert_anon"
    ON feedback FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);

DROP POLICY IF EXISTS "feedback_select_anon" ON feedback;
CREATE POLICY "feedback_select_anon"
    ON feedback FOR SELECT
    TO anon, authenticated
    USING (true);

-- ── query_log: just a timestamp per query, used for the "queries this month"
-- stat. Nothing sensitive; same pattern as feedback.
ALTER TABLE query_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "query_log_insert_anon" ON query_log;
CREATE POLICY "query_log_insert_anon"
    ON query_log FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);

DROP POLICY IF EXISTS "query_log_select_anon" ON query_log;
CREATE POLICY "query_log_select_anon"
    ON query_log FOR SELECT
    TO anon, authenticated
    USING (true);

-- ── paper_alerts: contains real subscriber email addresses — this is the one
-- table with actual PII. SELECT must NEVER be granted to anon/authenticated;
-- only the service-role key (send_alerts.py's weekly cron) may read it, and
-- that key bypasses RLS by design, so it needs no policy here at all.
--
-- The app subscribes users via an upsert on the `email` unique constraint, so
-- both INSERT and UPDATE are needed. Known limitation this does NOT fix: since
-- there's no auth system, anyone who knows (or guesses) another person's email
-- can overwrite that person's subscribed topics via the same upsert path. That
-- is a pre-existing product-design gap, not something RLS can close without
-- adding real authentication — flagging it rather than silently leaving it
-- undocumented.
ALTER TABLE paper_alerts ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "paper_alerts_insert_anon" ON paper_alerts;
CREATE POLICY "paper_alerts_insert_anon"
    ON paper_alerts FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);

DROP POLICY IF EXISTS "paper_alerts_update_anon" ON paper_alerts;
CREATE POLICY "paper_alerts_update_anon"
    ON paper_alerts FOR UPDATE
    TO anon, authenticated
    USING (true)
    WITH CHECK (true);
-- Deliberately no SELECT policy here — this is what keeps the email list
-- private from the anon/publishable key.

-- ═══════════════════════════════════════════════════════════════════════════
-- Storage bloat fix — duplicate chunks from non-idempotent batch retries
-- ═══════════════════════════════════════════════════════════════════════════
-- Root cause: process_and_load() in etl_pipeline.py dedups once per PAPER
-- (checks metadata->>'url' before starting) but then inserts 10-row batches
-- via insert_documents() with a bare "retry on any exception" loop. That
-- function is SECURITY DEFINER with EXECUTE still granted to PUBLIC (so anon
-- could call it directly too — see the REVOKE below) and had no uniqueness
-- constraint, so a retried batch — or a paper reprocessed because the
-- per-paper existence check didn't catch it — just inserted the same rows
-- again. Confirmed on the live project (2026-09-16): 20,855 of 171,464 rows
-- were exact duplicates.
--
-- APPLIED STATUS: this section has already been run against the production
-- project (bweglnxbuoumxnhrqhak) via the Supabase MCP connector. It's kept
-- here as the source of truth for what was done / to replicate elsewhere —
-- not as a "run this" script anymore. Note two things that differ from a
-- naive read of the SQL below:
--   1. A GENERATED ALWAYS ... STORED column requires a full table rewrite,
--      which timed out repeatedly against this project. We used a plain
--      expression-based unique index instead (no extra column, no backfill
--      needed) — functionally equivalent, simpler, faster to build.
--   2. REVOKE EXECUTE ... FROM anon, authenticated is NOT enough — Postgres
--      grants EXECUTE to PUBLIC by default at function creation, and every
--      role inherits PUBLIC's grants regardless of role-specific revokes.
--      You must REVOKE ... FROM PUBLIC explicitly, and re-run it after any
--      CREATE OR REPLACE FUNCTION, since replacing a function resets its
--      ACL back to the default (EXECUTE granted to PUBLIC) even though the
--      function's own name/signature didn't change.

-- Step 0: close the anon-callable write hole. Do this FIRST, independent of
-- everything else below — it's a pure permission change, zero data risk.
REVOKE EXECUTE ON FUNCTION public.insert_documents(jsonb) FROM PUBLIC;

-- Step 1: snapshot the rows about to be deleted, as a cheap safety net.
CREATE TABLE IF NOT EXISTS documents_dupes_backup AS
SELECT a.*
FROM documents a
JOIN (
    SELECT content, metadata->>'url' AS url, min(id) AS keep_id
    FROM documents
    GROUP BY content, metadata->>'url'
    HAVING count(*) > 1
) x ON a.content = x.content AND a.metadata->>'url' = x.url
WHERE a.id <> x.keep_id;
-- Verify: SELECT count(*) FROM documents_dupes_backup;  -- should match the
-- diagnostic duplicate_rows count from further up this file before proceeding.

-- Step 2: remove exact duplicate rows, keeping the lowest id per
-- (content, url) pair. NOTE: on the live project this DELETE was
-- consistently blocked by the harness's auto-mode classifier for bulk
-- writes — it had to be run by the user directly in the Supabase SQL editor.
DELETE FROM documents a
USING documents b
WHERE a.id > b.id
  AND a.content = b.content
  AND a.metadata->>'url' = b.metadata->>'url';

-- Step 3: prevent this from ever recurring — a unique index on the same
-- (content, url) fingerprint, computed as an expression (no extra column,
-- no backfill required). content can exceed btree's ~2.7KB row limit
-- (observed max 11.6KB on live data), so hash it rather than indexing raw
-- content directly.
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_fingerprint
    ON documents (md5(content || (metadata->>'url')));

-- Step 4: reclaim the disk space the deleted rows leave behind. Cannot run
-- inside a transaction block — if your tooling wraps statements in one
-- (ours did), run this one on its own, e.g. directly in the SQL editor.
-- Expect this to take a while (~90s+ for ~150K rows on the free tier) and to
-- hold a brief exclusive lock — any concurrent query against `documents`
-- waits until it completes rather than failing outright.
VACUUM (FULL, ANALYZE) documents;

-- Step 5: make insert_documents() itself idempotent, so a retried/duplicate
-- batch becomes a silent no-op. This is the ACTUAL definition that was live
-- on the project before patching — SECURITY DEFINER and the statement_timeout
-- are both original, only the ON CONFLICT clause is new. Remember: this
-- CREATE OR REPLACE resets the function's EXECUTE grant to PUBLIC, so
-- Step 0's REVOKE must be re-run immediately after this.
CREATE OR REPLACE FUNCTION public.insert_documents(rows jsonb)
 RETURNS void
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
BEGIN
  SET LOCAL statement_timeout = '60s';

  INSERT INTO public.documents (content, embedding, metadata)
  SELECT
    r->>'content',
    (r->>'embedding')::vector,
    (r->>'metadata')::jsonb
  FROM jsonb_array_elements(rows) AS r
  ON CONFLICT (md5(content || (metadata->>'url'))) DO NOTHING;
END;
$function$;

REVOKE EXECUTE ON FUNCTION public.insert_documents(jsonb) FROM PUBLIC;

-- ═══════════════════════════════════════════════════════════════════════════
-- Optional further space saving — halve embedding storage
-- ═══════════════════════════════════════════════════════════════════════════
-- Requires pgvector >= 0.7.0 (check with: SELECT extversion FROM pg_extension
-- WHERE extname = 'vector';). halfvec stores each dimension in 2 bytes instead
-- of 4, roughly halving the ~1.5KB/row the embedding column currently costs,
-- with a typically negligible recall impact for retrieval at this corpus size.
-- This requires updating retrieve_documents()/hybrid_search/match_documents
-- to cast query vectors to halfvec(384) too — don't run this half without the
-- matching app-side change, or search will break.
--
-- ALTER TABLE documents
--     ALTER COLUMN embedding TYPE halfvec(384) USING embedding::halfvec(384);
