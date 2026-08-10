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
sharing -- so treat the Kaggle Secret holding it like you would any credential
with full Drive access, and revoke/re-authorize it (Google Account ->
Security -> Third-party access) if you ever suspect it leaked.

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

This produces a `[gdrive]` section in your local `rclone.conf` with
`client_id`, `client_secret`, `scope`, and a `token` (containing
`access_token` + `refresh_token`). rclone auto-refreshes the access token
using the refresh token, so this keeps working across many Kaggle sessions
without re-authorizing -- as long as you don't revoke access.

## 2. Create a Drive folder for checkpoints

In Google Drive, create a folder (e.g. `nanochat-checkpoints`) and copy its
ID from the URL: `https://drive.google.com/drive/folders/<FOLDER_ID>`.

## 3. Store as Kaggle Secrets

In the Kaggle notebook editor: Add-ons -> Secrets.

1. `GDRIVE_OAUTH_CONF` -- from your local `rclone.conf`'s `[gdrive]` section,
   copy just the `client_id`, `client_secret`, and `token` lines (3 lines) as
   the secret value. Don't include `type =` or `scope =` -- the notebook adds
   those itself.
2. `GDRIVE_FOLDER_ID` -- the folder ID from step 2.

Attach both to the notebook (toggle "Attached") before running it.

Never commit `rclone.conf` or these values to git -- both are excluded in
`.gitignore` (`rclone.conf`, `*credentials*.json`).

## 4. What the notebook does with them (Cell 2)

```bash
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf <<EOF
[gdrive]
type = drive
scope = drive
<contents of GDRIVE_OAUTH_CONF secret>
root_folder_id = <GDRIVE_FOLDER_ID secret>
team_drive =
EOF

rclone lsd gdrive:   # sanity check -- should list folders (or be empty), not an auth error
```

## 5. Layout on Drive

Everything lives under the folder from step 2 (`gdrive:` root, since
`root_folder_id` already scopes it), mirroring `$NANOCHAT_BASE_DIR` locally:

```
gdrive:
  tokenizer/                     # trained BPE tokenizer + token_bytes.pt
  base_data_climbmix/            # (optional) cached pretraining shards, see kaggle_train.ipynb Cell 3
  base_checkpoints/d4/           # real pretraining checkpoints (model_/optim_/meta_ per step)
  base_checkpoints/d4-smoketest/ # SMOKE_TEST checkpoints -- separate tag, never resumed into the real run
  chatsft_checkpoints/d4/        # SFT checkpoints
```
