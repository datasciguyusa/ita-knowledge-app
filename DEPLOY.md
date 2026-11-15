# Putting the Tennis Rules app online (GitHub + Streamlit Community Cloud)

Run the commands in VS Code's terminal (Terminal → New Terminal), inside the app folder.

## 1. Check it works on your PC
1. `.env` has `ANTHROPIC_API_KEY=…` and `ADMIN_EMAILS=your@email.com`
   (if your key starts with `sk-ant-usr`, also add `ANTHROPIC_WORKSPACE_ID=…`, or use a workspace key)
2. `python check_claude.py` prints **SUCCESS**
3. `streamlit run app.py` → sign in → open **Test prep** once (reads every practice test into
   `qa_cache/`, so the web app doesn't pay to do it again) → close the app (Ctrl+C)

## 2. Bundle the documents
`python sync_docs.py` — copies the White Badge Pre-Reqs folder into `docs/` (the web app can't see your PC).

## 3. Make a free database (keeps spending totals + approvals when the web app restarts)
1. Sign up at **neon.tech** (free) → create a project → copy the **connection string**
   (looks like `postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require`)
2. Change the start from `postgresql://` to `postgresql+psycopg://` — you'll paste it in step 5

## 4. Upload to GitHub (PRIVATE repository)
Easiest: VS Code **Source Control** panel → type a message → **Commit** (choose "Yes" to stage all) →
**Publish Branch** → **Publish to GitHub private repository**.

Or in the terminal, after creating an empty **private** repo on github.com:
```
git add -A
git commit -m "Tennis app: test prep, login, spending limit"
git remote add origin https://github.com/YOUR-NAME/tennis-knowledge-app.git
git push -u origin main
```
Must be private: it contains the ITF documents and exam papers. `.env` is never uploaded.

## 5. Deploy on Streamlit Community Cloud
1. **share.streamlit.io** → sign in with GitHub → **Create app** → deploy from GitHub
2. Repository: your repo · Branch: `main` · Main file: `app.py` · pick an App URL
3. **Advanced settings** → Python 3.12 → **Secrets**, paste and fill in:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   # ANTHROPIC_WORKSPACE_ID = "..."   # only if your key starts with sk-ant-usr
   ADMIN_EMAILS = "your@email.com"
   USAGE_DB_URL = "postgresql+psycopg://..."
   MONTHLY_LIMIT_USD = "3"
   ```
4. **Deploy** (first start takes a few minutes)

## 6. Let people in
A private repo makes a private app, so each person needs two things:
1. **Invite** them: app → **Share** → enter email → Invite (they sign in with Google or an emailed link)
2. **Approve** them: add their email to `allowed_emails.txt` (commit + push), or click **Approve**
   in the app's **🔐 Manage access** after they first sign in

Optional, one list instead of two: set up Google sign-in (below), then make the app public
(Settings → Sharing). The app's approval list then does all the gatekeeping.

## Updating later
Edit code or `allowed_emails.txt` → commit → push. If documents changed, run `python sync_docs.py`
(and open Test prep once locally for new tests) before committing. Streamlit redeploys automatically.

## Spending limit
Each email gets **$3/month** (change with `MONTHLY_LIMIT_USD`), resetting on the 1st, Mountain time.
Admins have no limit and see everyone's spending in the sidebar.

## Optional: verified Google sign-in
Without it, people type their email, so someone could type another approved person's address.
1. Google Cloud Console → APIs & Services → Credentials → Create OAuth client ID → Web application.
   Authorised redirect URI: `https://YOUR-APP.streamlit.app/oauth2callback`
2. Add to the Secrets:
   ```
   [auth]
   redirect_uri = "https://YOUR-APP.streamlit.app/oauth2callback"
   cookie_secret = "any long random string"
   client_id = "…from Google…"
   client_secret = "…from Google…"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```
The app shows a "Sign in with Google" button automatically once `[auth]` is present.
