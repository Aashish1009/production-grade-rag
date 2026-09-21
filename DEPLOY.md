# Deploying

How this app is published. Read this before changing anything about hosting.

## The live link

```
https://laptop-q2f9a3ff.tail7fd95f.ts.net
```

Stable. It is tied to the **machine's identity**, not to a running process, so
restarting the app mints no new hostname — the reason Funnel won over a
Cloudflare quick tunnel, which produced a new link per process. It is the *URL*
that is stable, not the reachability. Two things have to hold for the link to
answer at all:

1. **The laptop must be awake, with a signed-in session.** This is the only copy
   of the app; it is not mirrored anywhere. Sleep, shutdown, or a locked screen
   with no session takes the link down.
2. **The supervisor must be alive.** `deploy/run-rag.ps1` keeps both the app and
   the tunnel up (10 s watchdog, restarts either). It is registered as a task
   that fires at **logon** — `LogonType Interactive`, 45 s delay — not at boot,
   so a reboot that stops at the sign-in screen leaves the link dead until
   someone signs in.

**Live and verified as of 2026-09-18.** TLS certificate issued by Let's Encrypt,
valid to 2026-12-17. Reachability was confirmed from *outside* the tailnet — not
merely from this host, which resolves the MagicDNS name to `100.98.244.99` over
the local Tailscale interface and would appear to work even if public ingress
were broken.

**This has already failed once.** On 2026-09-21 the `RAG Supervisor` task was
found `Ready` rather than `Running`: its supervisor had been gone since
2026-09-18 15:52 (`LastTaskResult 3221225786`, i.e. `STATUS_CONTROL_C_EXIT` —
killed, not crashed) and nothing had restarted it, because only a logon does. The
URL answered the whole time, because the app process itself had survived. A dead
supervisor is invisible until the thing it watches dies too, so trust this check
rather than the URL:

```powershell
.\deploy\run-rag.ps1 -Check

  rag-server   : UP  (127.0.0.1:8000)
  funnel       : ON
  public URL   : https://laptop-q2f9a3ff.tail7fd95f.ts.net
  reachable    : YES (HTTP 200)
```

Read `-Check`, not the task's state: a supervisor started by hand (a manual
`-Restart`, as on 2026-09-21) leaves the task showing `Ready` while the link is
perfectly served. After any reboot or logoff, sign in and wait 45 s.

## The constraints that drive every decision here

1. **$0, permanently. No credit card, ever.** This is not "prefer free" — it is
   hard. It disqualifies Oracle, Fly, Koyeb, Railway, Cloud Run, AWS, GCP and
   Azure at signup, however generous their free tiers are.
2. **The link must work when someone opens it.** It goes on a resume; a dead link
   is worse than no link.
3. **No login wall.** Anyone with the URL can upload, query, and delete.

## Why this app cannot run on an ordinary free tier

Two independent walls. Either one alone is fatal.

**Wall 1 — it needs ~1.0 GB of committed memory before a model loads.** The idle
process reports a misleading 42 MB because the models are lazy-loaded, but the
*imports* are not: `import torch` and `import sentence_transformers` commit memory
that is never handed back to the OS for the life of the process. Measured on this
machine 2026-09-21, project venv, nothing touching `.model` (private = commit
charge, the figure a host must be able to allocate; working set = resident):

```
                                 private              working set
fresh interpreter                    8 MB                   14 MB
+ import torch                     402 MB    (+394)       195 MB
+ sentence_transformers            867 MB    (+465)       450 MB
+ langchain_huggingface            888 MB                  473 MB
+ langchain_qdrant                 931 MB                  521 MB
```

That is the floor with **no model loaded**. The weights themselves are ~230 MB on
top of it, which the live server's own counters confirm:

```
live server, models resident     1,280 MB                  484 MB
live server, models released     1,051 MB    (-229)        425 MB
peak after serving questions     1,589 MB        (earlier process)
```

An earlier version of this file said ~700–900 MB. That estimate was low, and it
was stated without a metric: it summed the libraries and the weights, which is
working-set reasoning, and then compared the total against a private-memory
limit. The error ran the wrong way. A 512 MB tier fails at `import torch`
(~400 MB) and dies well before a question is asked, so every free tier is out by
a *wider* margin than this file used to claim.

Render's free tier gives 512 MB. This app crosses that line *while importing its
libraries*, before a single model loads — it would OOM on boot, not on the first
question. PythonAnywhere's 512 MB has the same ceiling, plus it blocks outbound
sockets, which kills both Groq and Tavily.

**Wall 2 — it needs a persistent disk.** Qdrant runs *embedded*
(`QdrantClient(path=...)`, see `src/rag/indexing.py`), so the entire vector store
is a directory of files on local disk. Free tiers wipe their disks on every
restart and idle spin-down. The app would boot, serve an empty UI, and answer
nothing.

**Moving the store to Qdrant Cloud fixes Wall 2 and not Wall 1.** It is worth
doing for durability and portability — see below — but it does not unlock free
hosting. There is no free, card-free host with ~1 GB of RAM that runs an
arbitrary Python process.

Fitting inside 512 MB would mean replacing torch with ONNX, dropping the
cross-encoder reranker, and dropping docling/surya ingestion. That is a different,
lesser app — and it would still be stuck on Render's 0.1 CPU.

## The models are dropped when idle

The link is idle almost all of the time — measured over the first 66 hours it
served **two visits**. Holding 1.3 GB of models for a link nobody is reading is
the waste the hosting search was about, so both models (dense embedder,
cross-encoder) are released after `RAG_MODEL_IDLE_UNLOAD_MINUTES` (default 10)
with no use, and rebuilt on demand.

It recovers **~229 MB, not the whole footprint** — 1,280 MB private becomes
1,051 MB. The ~1.0 GB of imports above is the floor and cannot be returned to the
OS from a live process; only exiting frees it. That is still the right trade
against a cold load landing inside a visitor's first question, but do not read
this feature as "the app costs nothing while idle".

**What counts as use:** an embed, a rerank, or a warm-up. A page load POSTs
`/api/warm`, and even a no-op warm-up enters the guarded block, so each visit
pushes the release ten minutes out — deliberate, since someone who just opened
the page is about to ask something. The 10 s `/api/stats` poll is *not* use: a
release was observed firing on schedule (`released dense embedder and reranker
after 10.3 min idle`) while a browser polled throughout. Because a visit defers
the release silently, the release line in `logs/server.log` — not the absence of
one — is the only evidence the reaper is working.

**The visitor is told.** `GET /api/stats` reports `models_loaded`, the page calls
`POST /api/warm` on open, and while that runs the UI shows a "Loading the models"
notice with elapsed seconds and says the wait is one-time. A document uploaded
during the load is queued and starts indexing when it finishes. Measured warm-up
on this machine: **8.2 s dense + 8.9 s reranker ≈ 17 s** with the weights in the
OS page cache; ~90 s on a cold cache (immediately after a reboot).

Set `RAG_MODEL_IDLE_UNLOAD_MINUTES=0` to disable unloading and keep them resident.

## The one-time Tailscale setup

Tailscale Funnel is available on the free plan, needs no card, requires no owned
domain, and shows no browser interstitial warning.

1. **Install** (already done here, v1.102.4):
   ```powershell
   winget install --id Tailscale.Tailscale -e
   ```
2. **Sign in.** Already done on this machine. Verify:
   ```powershell
   & "C:\Program Files\Tailscale\tailscale.exe" status
   ```
3. **Enable HTTPS certificates** — <https://login.tailscale.com/admin/dns>
   → *HTTPS Certificates* → **Enable HTTPS**
4. **Grant the Funnel permission.** The console's *visual* Access Controls editor
   has **no Funnel control** — it only exists in the JSON editor
   (<https://login.tailscale.com/admin/acls> → left sidebar → **JSON editor**).
   Add this block at the top level, alongside `grants`:

   ```json
   "nodeAttrs": [
       {
           "target": ["autogroup:member"],
           "attr": ["funnel"],
       },
   ],
   ```

   The complete working policy is saved at `deploy/tailscale-policy.json`.

Steps 3 and 4 are mandatory and easy to miss. **Without them `tailscale funnel`
prints nothing at all and hangs indefinitely** waiting on an interactive approval
that never arrives. That is a silent hang, not an error — do not read it as
"still working". An instant return is the success signal.

## Bringing it up

```powershell
# 1. the app itself, on 127.0.0.1:8000
.\deploy\run-rag.ps1

# 2. publish it (one time; persists across reboots thanks to --bg)
& "C:\Program Files\Tailscale\tailscale.exe" funnel --bg 8000

# 3. verify
& "C:\Program Files\Tailscale\tailscale.exe" funnel status
```

Funnel listens publicly only on ports **443**, **8443** and **10000**, and is
TLS-only.

**Launch the server from PowerShell, never Git Bash.** Under MSYS, `hi_res` PDF
parsing aborts the process with a Cygwin `TP_NUM_C_BUFS` fatal error. Also check
that Qdrant's directory lock is free before starting a second instance.

## Optional: Qdrant Cloud

The free tier at <https://cloud.qdrant.io> is creatable **without a card**
(email / Google / GitHub): 0.5 vCPU, 1 GB RAM, 4 GB disk, "free forever", one
cluster per account.

Pointing the app at it removes the persistent-disk requirement and makes the app
portable to any future host. It requires a small code change — the code is
currently hardwired to embedded mode:

- `src/rag/config.py:72` — add `qdrant_url` and `qdrant_api_key` settings
- `src/rag/indexing.py:124` — branch on them:
  `QdrantClient(url=..., api_key=...)` when set, `QdrantClient(path=...)` otherwise

Keep it backward-compatible: with no URL set, behaviour must be byte-identical to
today.

Caveat: Qdrant suspends free clusters after inactivity, so it may need waking
from the dashboard.

## What `deploy/` contains

It is the **hosting kit, not part of the app**. The app lives in `src/`,
`frontend/`, `data/` and `qdrant_db/`.

| Item | Purpose | Once Funnel is live |
|---|---|---|
| `run-rag.ps1` | supervisor: starts the app + tunnel, auto-restarts on crash | **keep** — still guards the app |
| `bin/cloudflared.exe` | mints the ephemeral quick-tunnel URL | **now redundant** — removable |
| `current-url.txt` | where the quick-tunnel URL is written | **now redundant** — removable |
| `logs/` | runtime logs | harmless |
| `tailscale-policy.json` | the working tailnet ACL policy, for reference/recovery | keep |

## Two lessons already paid for

**A Cloudflare quick tunnel is not a deployment.** It mints a random hostname per
process and retires the old one permanently, so every restart produces a new link
and every old link becomes a Cloudflare **error 1033**. This project's original
link died exactly that way. Never put a quick-tunnel URL on anything durable.

**`cloudflared` logs to stderr, not stdout.** A parser reading only stdout
concludes "no URL appeared" while the tunnel is perfectly healthy. Read both
streams.
