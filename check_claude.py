"""Check that this PC can reach Claude and that the API key works:   python check_claude.py"""
import os
import socket
import ssl
import sys
import urllib.request

from dotenv import load_dotenv

HOST = "api.anthropic.com"
load_dotenv()
key = os.getenv("ANTHROPIC_API_KEY", "")
ws = os.getenv("ANTHROPIC_WORKSPACE_ID", "")


def step(name, fn):
    try:
        result = fn()
        print(f"[ OK ] {name}" + (f" — {result}" if result else ""))
        return True
    except Exception as exc:
        print(f"[FAIL] {name} — {type(exc).__name__}: {exc}")
        return False


print("Python:", sys.version.split()[0], "|", sys.executable)
import importlib.metadata as md
import re
for pkg in ("anthropic", "httpx", "certifi", "truststore"):
    try:
        print(f"{pkg} {md.version(pkg)}", end=" | ")
    except md.PackageNotFoundError:
        print(f"{pkg} -", end=" | ")
print()
for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
    if os.getenv(var):
        print(f"env {var} = {re.sub(r'//[^@/]+@', '//***@', os.getenv(var))}")   # hide any password
print("key found:", bool(key), "| starts with:", (key[:10] + "…") if key else "-", "| workspace ID:", ws or "(none)")
print()

ok = step("1. Look up api.anthropic.com (DNS)", lambda: socket.gethostbyname(HOST))
ok = ok and step("2. Open a connection on port 443", lambda: socket.create_connection((HOST, 443), timeout=10).close())


def tls():
    ctx = ssl.create_default_context()
    with socket.create_connection((HOST, 443), timeout=10) as s, ctx.wrap_socket(s, server_hostname=HOST) as t:
        return "issuer: " + dict(x[0] for x in t.getpeercert()["issuer"]).get("organizationName", "?")


def tls_certifi():
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    with socket.create_connection((HOST, 443), timeout=10) as s, ctx.wrap_socket(s, server_hostname=HOST):
        return "ok with certifi's certificates"


ok = ok and (step("3. Secure (TLS) handshake, Windows certificates", tls) | step("3b. Secure handshake, certifi certificates", tls_certifi))


def https_get():
    try:
        urllib.request.urlopen(f"https://{HOST}/v1/models", timeout=15)
    except urllib.error.HTTPError as e:
        return f"server answered HTTP {e.code} (expected without a key)"


ok = ok and step("4. HTTPS request", https_get)

print()
try:
    import anthropic
    client = anthropic.Anthropic(api_key=key, default_headers={"anthropic-workspace-id": ws} if ws else None)
    r = client.messages.create(model="claude-haiku-5-5", max_tokens=20,
                               messages=[{"role": "user", "content": "Reply with just: OK"}])
    print("SUCCESS — Claude replied:", r.content[0].text)
except Exception as exc:
    print("FAILED:", type(exc).__name__, "-", exc)
    cause = exc.__cause__ or exc.__context__
    depth = 0
    while cause is not None and depth < 5:
        print("   caused by:", type(cause).__name__, "-", cause)
        cause = cause.__cause__ or cause.__context__
        depth += 1

print()
db = os.getenv("USAGE_DB_URL", "")
if not db:
    print("Database: USAGE_DB_URL not set (the app will use a local usage.db file)")
else:
    try:
        from usage import AccessList, UsageStore
        store = UsageStore(db)
        AccessList(store)
        print("Database: SUCCESS — connected and tables ready |", store.engine.dialect.name,
              "| users with spending this month:", len(store.all_users()))
    except Exception as exc:
        print("Database: FAILED —", type(exc).__name__, "-", str(exc)[:300])
