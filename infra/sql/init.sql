create table if not exists prediction_logs (
  id bigserial primary key,
  request_id uuid not null,
  comment_text text not null,
  input_hash char(64) not null,
  scores jsonb not null,           -- e.g. {"toxic":0.83,"insult":0.41,...}
  labels text[] not null,          -- e.g. {"toxic","insult"}
  model_name text not null,        -- e.g. "toxic-comment"
  model_version text not null,     -- e.g. "3" or "Production@3"
  latency_ms integer not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_prediction_logs_created_at on prediction_logs(created_at);
create index if not exists idx_prediction_logs_input_hash on prediction_logs(input_hash);

create table if not exists feedback (
  id bigserial primary key,
  request_id uuid not null references prediction_logs(request_id) on delete cascade,
  correct boolean not null,        -- overall correctness toggle
  true_labels text[] null,         -- optional: user-specified true labels
  notes text null,
  created_at timestamptz not null default now()
);