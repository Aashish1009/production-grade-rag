# Deploying

How this app is published. Read this before changing anything about hosting.

## The live link

```
https://laptop-q2f9a3ff.tail7fd95f.ts.net
```

Stable. It is tied to the **machine's identity**, not to a running process, so it
survives reboots, crashes, and restarts unchanged.

**Live and verified as of 2026-09-18.** TLS certificate issued by Let's Encrypt,
valid to 2026-12-17. Reachability was confirmed from *outside* the tailnet — not
merely from this host, which resolves the MagicDNS name to `100.98.244.99` over
the local Tailscale interface and would appear to work even if public ingress
were broken.

## The constraints that drive every decision here

1. **$0, permanently. No credit card, ever.** This is not "prefer free" — it is
   hard. It disqualifies Oracle, Fly, Koyeb, Railway, Cloud Run, AWS, GCP and
   Azure at signup, however generous their free tiers are.
2. **The link must work when someone opens it.** It goes on a resume; a dead link
   is worse than no link.
3. **No login wall.** Anyone with the URL can upload, query, and delete.

## Why this app cannot run on an ordinary free tier

Two independent walls. Either one alone is fatal.

**Wall 1 — it needs ~700–900 MB of RAM.** The idle process reports a misleading
42 MB because models are lazy-loaded. Measured:

```
baseline python                      18 MB
after import torch                  204 MB   (+186)
after import sentence_transformers  470 MB   (+266)
+ bge-small-en-v1.5 weights         128 MB
+ cross-encoder reranker weights     88 MB
──────────────────────────────────────────────
realistic serving footprint      ~700–900 MB
```

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
