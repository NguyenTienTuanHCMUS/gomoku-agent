-- Table for games played on the web page. Run once in Supabase: SQL Editor -> New query -> paste -> Run.
--
-- Security model: the web page uses the public (publishable / anon) key, which anyone can read from the
-- page source. With row level security below, that key can only INSERT a well-formed game.
-- It cannot read, change or delete anything. You read the games from the Supabase dashboard,
-- or with fetch_games.py using the secret key, which must never be put in the page or the repo.

create table if not exists public.games (
  id          bigint generated always as identity primary key,
  created_at  timestamptz not null default now(),
  game_uid    uuid        not null,                       -- same uid if a game is saved again after "undo"
  version     smallint    not null default 1,
  result      text        not null check (result in ('human', 'model', 'draw', 'unfinished')),
  human       text        not null check (human in ('X', 'O')),
  sims        integer     not null check (sims between 0 and 100000),
  undos       integer     not null default 0 check (undos between 0 and 1000),
  model       text        not null check (char_length(model) between 1 and 32),
  moves       text[]      not null check (
                cardinality(moves) between 1 and 225
                and array_to_string(moves, ' ') ~ '^[a-o](1[0-5]|[1-9])( [a-o](1[0-5]|[1-9]))*$'
              )
);

alter table public.games enable row level security;

drop policy if exists "web page can add games" on public.games;
create policy "web page can add games" on public.games
  for insert to anon, authenticated
  with check (true);
-- No select / update / delete policies: with RLS on, the public key cannot do any of those.

-- Explicit grants, so this works whether or not "Grant API access to new tables automatically" was ticked.
grant usage on schema public to anon, authenticated;
revoke all on public.games from anon, authenticated;
grant insert on public.games to anon, authenticated;
