# Google Drive checkpoint sync (rclone + service account)

Kaggle sessions can die at any point (not just the 12h hard limit), and the
Kaggle filesystem is wiped between sessions. Checkpoints are synced to Google
Drive continuously during training so a killed session never loses more than
one `--save-every` interval of progress.

We use a **service account**, not a personal OAuth login, so the credential
stored in Kaggle Secrets is scoped to a single shared folder instead of your
entire Drive. If a Kaggle Secret ever leaked, a service account key only
exposes that one folder; a personal OAuth token (`scope=drive`) exposes
everything in your Drive.

## 1. Create a service account (Google Cloud Console)

1. Go to https://console.cloud.google.com/ and create a project (or reuse one).
2. APIs & Services -> Library -> enable **Google Drive API**.
3. APIs & Services -> Credentials -> Create Credentials -> **Service account**.
   - Name it something like `nanochat-kaggle-sync`. No project roles needed
     (Drive access is granted later via folder sharing, not IAM roles).
4. Open the created service account -> **Keys** tab -> Add Key -> Create new key
   -> **JSON**. This downloads a `.json` file — treat it like a password.
   It contains a private key with no expiry until you revoke it.

## 2. Share a Drive folder with the service account

Service accounts have no personal Drive storage quota of their own, so they
need a folder shared with them from your account.

1. In Google Drive, create a folder, e.g. `nanochat-checkpoints`.
2. Right-click -> Share -> paste the service account's `client_email` field
   from the JSON (looks like `nanochat-kaggle-sync@<project>.iam.gserviceaccount.com`).
   Give it **Editor** access.
3. Open the folder in the browser and copy the folder ID from the URL:
   `https://drive.google.com/drive/folders/<FOLDER_ID>`.

## 3. Store the JSON as a Kaggle Secret

1. In the Kaggle notebook editor: Add-ons -> Secrets.
2. Add a secret named `GDRIVE_SERVICE_ACCOUNT_JSON` — paste the **entire
   contents** of the downloaded JSON key file as the value.
3. Add a second secret named `GDRIVE_FOLDER_ID` — the folder ID from step 2.3.
4. Attach both secrets to the notebook (toggle "Attached" in the Secrets panel)
   before running it.

Never commit the JSON key or `rclone.conf` to git — both are excluded in
`.gitignore` (`*service-account*.json`, `rclone.conf`).

## 4. Building rclone.conf inside the Kaggle notebook

This is what notebook Cell 2 does automatically (see `kaggle/kaggle_train.ipynb`),
shown here for reference / local testing:

```bash
mkdir -p ~/.config/rclone
cat > ~/gdrive-sa.json <<'EOF'
<contents of GDRIVE_SERVICE_ACCOUNT_JSON secret>
EOF

cat > ~/.config/rclone/rclone.conf <<EOF
[gdrive]
type = drive
scope = drive
service_account_file = /root/gdrive-sa.json
root_folder_id = <GDRIVE_FOLDER_ID secret>
team_drive =
EOF

rclone lsd gdrive:   # sanity check — should list nothing or existing subfolders, no auth error
```

## 5. Layout on Drive

Everything lives under the shared folder (`gdrive:` root, since `root_folder_id`
already scopes it), mirroring `$NANOCHAT_BASE_DIR` locally:

```
gdrive:
  tokenizer/                     # trained BPE tokenizer + token_bytes.pt
  base_data_climbmix/            # (optional) cached pretraining shards, see kaggle_train.ipynb Cell 3
  base_checkpoints/d4/           # pretraining checkpoints (model_/optim_/meta_ per step)
  chatsft_checkpoints/d4/        # SFT checkpoints
```

## Local fallback (not recommended for Kaggle Secrets)

If you already have a personal rclone Google Drive remote configured locally
(`type = drive`, OAuth `client_id`/`client_secret`/`token`, no
`service_account_file`), it will work the same way functionally, but the
Kaggle Secret would then hold OAuth credentials with full access to your
entire Drive (`scope = drive`), not just one folder. Only use this as a
quick/throwaway option, and prefer rotating those credentials afterward if
you do.
