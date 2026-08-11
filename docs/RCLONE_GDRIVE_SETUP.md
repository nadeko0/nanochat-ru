# Google Drive checkpoint sync (rclone)

Kaggle sessions can die at any point (not just the 12h hard limit), and the
Kaggle filesystem is wiped between sessions. Checkpoints are synced to Google
Drive continuously during training so a killed session never loses more than
one `--save-every` interval of progress.

## Why not a service account

The original plan here was a service account scoped to a single shared
folder, so a leaked Kaggle Secret would only expose that folder, not the
whole Drive. That doesn't work on a personal (non-Workspace) Google account:
service accounts have no storage quota of their own, and creating files in a
folder shared with them fails with `storageQuotaExceeded`, even with Editor
access. Writing into someone else's quota via a shared folder only works on
a **Shared Drive** (pooled quota, no personal owner), and Shared Drives
require a Google Workspace subscription -- not available on a plain gmail.com
account. (Discovered this via the notebook's `SMOKE_TEST` run actually
failing with `storageQuotaExceeded` on the first real upload attempt.)

So instead this uses a **personal OAuth rclone remote** -- rclone
authenticated as you, writing files that you own, against your own 5TB
quota. The real trade-off: it has access to your whole Drive (`scope=drive`),
not just one folder. There's no way to scope a personal-account OAuth token
down to a single folder the way a service account key can be scoped by
sharing -- so treat the credentials in Kaggle Secrets like you would any
credential with full Drive access, and revoke/re-authorize them (Google
Account -> Security -> Third-party access) if you ever suspect they leaked
(e.g. pasted somewhere they shouldn't have been).

## 1. Generate the OAuth remote locally

If you already have an rclone Google Drive remote configured on your machine
(check `rclone config show` or look for `~/.config/rclone/rclone.conf` /
`%APPDATA%\rclone\rclone.conf` on Windows), you can reuse it -- skip to step 2.

Otherwise, install rclone locally and run:

```bash
rclone config
# n) New remote, name it "gdrive", type "drive" (Google Drive)
# leave client_id/client_secret blank to use rclone's own, or set your own
# scope: 1 (full access, "drive")
# leave root_folder_id / service_account_file blank
# "Use auto config?" -> yes -- this opens a browser to log in and authorize
```

This produces a `[gdrive]` section in your local `rclone.conf` with these
fields: `client_id`, `client_secret`, and a `token` (a JSON blob containing
`access_token` + `refresh_token`). rclone auto-refreshes the access token
using the refresh token, so this keeps working across many Kaggle sessions
without re-authorizing, as long as you don't revoke access.

## 2. Create a Drive folder for checkpoints

In Google Drive, create a folder (e.g. `nanochat-checkpoints`) and copy its
ID from the URL: `https://drive.google.com/drive/folders/<FOLDER_ID>`.

## 3. Store as Kaggle Secrets -- four separate single-line values

In the Kaggle notebook editor: Add-ons -> Secrets. Kaggle's Secrets box is a
single-line field: pasting a multi-line `rclone.conf` snippet into one
secret loses the newlines and everything gets mangled into one broken line.
Use four separate secrets instead, each just the bare value (no `key =`
prefix):

| Secret name | Value (from your local `rclone.conf`'s `[gdrive]` section) |
|---|---|
| `GDRIVE_CLIENT_ID` | the `client_id` value |
| `GDRIVE_CLIENT_SECRET` | the `client_secret` value |
| `GDRIVE_OAUTH_TOKEN` | the whole `token` value -- the `{"access_token":...}` JSON blob (already single-line, safe to paste as-is) |
| `GDRIVE_FOLDER_ID` | the folder ID from step 2 |

Attach all four to the notebook (toggle "Attached") before running it.

Never commit `rclone.conf` or these values to git -- both are excluded in
`.gitignore` (`rclone.conf`, `*credentials*.json`).

## 4. What the notebook does with them (Cell 2)

```bash
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf <<EOF
[gdrive]
type = drive
scope = drive
client_id = <GDRIVE_CLIENT_ID secret>
client_secret = <GDRIVE_CLIENT_SECRET secret>
token = <GDRIVE_OAUTH_TOKEN secret>
root_folder_id = <GDRIVE_FOLDER_ID secret>
team_drive =
EOF

rclone lsd gdrive:   # sanity check -- should list folders (or be empty), not an auth error
```

## 5. Layout on Drive

Everything lives under the folder from step 2 (`gdrive:` root, since
`root_folder_id` already scopes it), mirroring `$NANOCHAT_BASE_DIR` locally.
Per-model checkpoints are namespaced by `--model-tag` subfolder automatically
(`base_checkpoints/<tag>/`, `chatsft_checkpoints/<tag>/`), so different
models sharing the same `vocab_size=32768` tokenizer coexist safely:

```
gdrive:
  tokenizer/                     # shared BPE tokenizer (vocab_size=32768) -- d4, d6, a10 all load this
  tokenizer_a9/                  # SEPARATE tokenizer (vocab_size=16384) -- a9 only, see the gotcha below
  base_data_climbmix/            # cached pretraining shards, shared across all models/vocab sizes
  base_checkpoints/d4/           # d4 pretraining checkpoints (model_/optim_/meta_ per step)
  base_checkpoints/d4-smoketest/ # SMOKE_TEST checkpoints -- separate tag, never resumed into the real run
  base_checkpoints/d6/           # d6, a10, a9's base checkpoints similarly, one folder per --model-tag
  chatsft_checkpoints/d4/        # SFT checkpoints, same per-tag convention
```

**Gotcha, hit for real on A9** (see docs/RESEARCH_LOG.md 2026-08-11): the `tokenizer/` path is
**not** namespaced by model tag the way checkpoints are -- `nanochat/tokenizer.py` always writes
to `{NANOCHAT_BASE_DIR}/tokenizer`, and `kaggle/sync_checkpoints.py`'s background sync always
targets `gdrive:tokenizer`. Training a tokenizer with a *different* `vocab_size` (as A9 did,
16384 instead of the shared 32768) while pointed at the same `NANOCHAT_BASE_DIR`/sync setup as
an existing model would silently overwrite that model's tokenizer on Drive. The fix used for
A9: a separate `NANOCHAT_BASE_DIR` (so the new tokenizer trains into an isolated local path), a
manual one-time `rclone copy` to a distinct remote path (`gdrive:tokenizer_a9`, not
`gdrive:tokenizer`), and `sync_checkpoints.py --skip-subdirs tokenizer` for the background sync
during training (so it never touches `tokenizer/` at all). See
`vastai/vastai_train_a9.ipynb` Cells 2-3 for the working pattern if this project ever needs a
third vocab_size.
