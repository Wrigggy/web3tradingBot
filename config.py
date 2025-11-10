"""Runtime configuration for the momentum trading bot.

Populate the constants below with your live or paper credentials before
running the bot on AWS. The start script will read this file and export the
values as environment variables before invoking ``2.py``.
"""

# Roostoo API credentials
ROOSTOO_API_KEY = "your-roostoo-api-key"
ROOSTOO_SECRET_KEY = "your-roostoo-secret-key"

# Horus market data API credential
HORUS_API_KEY = "your-horus-api-key"

# Optional: default poll interval override (seconds)
# Leave as ``None`` to use the value hardcoded in ``2.py``
POLL_INTERVAL_SECONDS = None
