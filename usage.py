"""Per-user, per-calendar-month spending on Claude, with a monthly limit.

Costs are worked out from the token counts Claude returns with every response.
Totals are stored in a database: SQLite file by default (fine on your PC), or any
SQLAlchemy URL in the USAGE_DB_URL setting (use a hosted Postgres on the web — the
web host's disk is wiped on restart, which would reset everyone's totals).
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text

# Claude Haiku 5.5 prices, USD per million tokens. Prompts over 100K tokens cost 5×.
PRICE_IN, PRICE_OUT = 0.10, 0.50
LONG_PROMPT_TOKENS, LONG_MULTIPLIER = 100_000, 5.0


class BudgetExceeded(Exception):
    pass


def cost_of(usage) -> float:
    """USD cost of one response, from its `usage` (input/output/cache token counts)."""
    g = lambda name: int(getattr(usage, name, 0) or 0)
    cache_write, cache_read = g("cache_creation_input_tokens"), g("cache_read_input_tokens")
    prompt = g("input_tokens") + cache_write + cache_read
    mult = LONG_MULTIPLIER if prompt > LONG_PROMPT_TOKENS else 1.0
    usd = (g("input_tokens") * PRICE_IN + cache_write * PRICE_IN * 1.25
           + cache_read * PRICE_IN * 0.1 + g("output_tokens") * PRICE_OUT) / 1_000_000
    return usd * mult


class UsageStore:
    def __init__(self, url: str, tz: str = "America/Denver"):
        self.engine = create_engine(url, pool_pre_ping=True)
        self.tz = ZoneInfo(tz)
        with self.engine.begin() as c:
            c.execute(text("""CREATE TABLE IF NOT EXISTS usage_monthly (
                email VARCHAR(320) NOT NULL, month CHAR(7) NOT NULL,
                cost_usd FLOAT NOT NULL DEFAULT 0, input_tokens BIGINT NOT NULL DEFAULT 0,
                output_tokens BIGINT NOT NULL DEFAULT 0, calls INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (email, month))"""))
            c.execute(text("""CREATE TABLE IF NOT EXISTS usage_log (
                ts VARCHAR(32) NOT NULL, email VARCHAR(320) NOT NULL, kind VARCHAR(40),
                input_tokens BIGINT, output_tokens BIGINT, cost_usd FLOAT)"""))

    def month(self) -> str:
        return datetime.now(self.tz).strftime("%Y-%m")

    def month_name(self) -> str:
        return datetime.now(self.tz).strftime("%B")

    def spent(self, email: str) -> float:
        with self.engine.connect() as c:
            v = c.execute(text("SELECT cost_usd FROM usage_monthly WHERE email=:e AND month=:m"),
                          {"e": email, "m": self.month()}).scalar()
        return float(v or 0.0)

    def record(self, email: str, kind: str, usage) -> float:
        cost = cost_of(usage)
        i, o = int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)
        p = {"e": email, "m": self.month(), "c": cost, "i": i, "o": o}
        with self.engine.begin() as c:
            c.execute(text("""INSERT INTO usage_monthly (email, month, cost_usd, input_tokens, output_tokens, calls)
                VALUES (:e, :m, :c, :i, :o, 1)
                ON CONFLICT (email, month) DO UPDATE SET
                  cost_usd = usage_monthly.cost_usd + excluded.cost_usd,
                  input_tokens = usage_monthly.input_tokens + excluded.input_tokens,
                  output_tokens = usage_monthly.output_tokens + excluded.output_tokens,
                  calls = usage_monthly.calls + 1"""), p)
            c.execute(text("INSERT INTO usage_log VALUES (:ts, :e, :k, :i, :o, :c)"),
                      {**p, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "k": kind})
        return cost

    def all_users(self, month: str | None = None) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(text("""SELECT email, cost_usd, calls FROM usage_monthly
                                     WHERE month=:m ORDER BY cost_usd DESC"""),
                             {"m": month or self.month()}).mappings().all()
        return [dict(r) for r in rows]


class Budget:
    """Ties one signed-in user to the store and the monthly limit."""

    def __init__(self, store: UsageStore, email: str, limit_usd: float, unlimited: bool = False,
                 on_change=None):
        self.store, self.email, self.limit, self.unlimited = store, email, limit_usd, unlimited
        self.on_change = on_change

    @property
    def spent(self) -> float:
        return self.store.spent(self.email)

    @property
    def remaining(self) -> float:
        return max(self.limit - self.spent, 0.0)

    def check(self) -> None:
        if not self.unlimited and self.spent >= self.limit:
            raise BudgetExceeded(
                f"You've used your ${self.limit:.2f} for {self.store.month_name()}. It resets on the 1st of next month.")

    def record(self, kind: str, usage) -> None:
        self.store.record(self.email, kind, usage)
        if self.on_change:
            try:
                self.on_change()
            except Exception:
                pass


class AccessList:
    """Emails the owner has approved, plus sign-in attempts waiting for approval."""

    def __init__(self, store: UsageStore):
        self.engine = store.engine
        with self.engine.begin() as c:
            c.execute(text("""CREATE TABLE IF NOT EXISTS allowed_emails (
                email VARCHAR(320) PRIMARY KEY, added_by VARCHAR(320), added_at VARCHAR(32))"""))
            c.execute(text("""CREATE TABLE IF NOT EXISTS access_requests (
                email VARCHAR(320) PRIMARY KEY, requested_at VARCHAR(32))"""))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def is_allowed(self, email: str) -> bool:
        with self.engine.connect() as c:
            return c.execute(text("SELECT 1 FROM allowed_emails WHERE email=:e"),
                             {"e": email}).first() is not None

    def allow(self, email: str, by: str) -> None:
        with self.engine.begin() as c:
            c.execute(text("""INSERT INTO allowed_emails (email, added_by, added_at) VALUES (:e, :b, :t)
                              ON CONFLICT (email) DO NOTHING"""), {"e": email, "b": by, "t": self._now()})
            c.execute(text("DELETE FROM access_requests WHERE email=:e"), {"e": email})

    def revoke(self, email: str) -> None:
        with self.engine.begin() as c:
            c.execute(text("DELETE FROM allowed_emails WHERE email=:e"), {"e": email})

    def request(self, email: str) -> None:
        with self.engine.begin() as c:
            c.execute(text("""INSERT INTO access_requests (email, requested_at) VALUES (:e, :t)
                              ON CONFLICT (email) DO NOTHING"""), {"e": email, "t": self._now()})

    def dismiss(self, email: str) -> None:
        with self.engine.begin() as c:
            c.execute(text("DELETE FROM access_requests WHERE email=:e"), {"e": email})

    def allowed(self) -> list[str]:
        with self.engine.connect() as c:
            return [r[0] for r in c.execute(text("SELECT email FROM allowed_emails ORDER BY email"))]

    def pending(self) -> list[str]:
        with self.engine.connect() as c:
            return [r[0] for r in c.execute(text("SELECT email FROM access_requests ORDER BY requested_at"))]
