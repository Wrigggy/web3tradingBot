"""Runtime configuration for the momentum trading bot.

Populate the constants below with your live or paper credentials before
running the bot on AWS. The start script will read this file and export the
values as environment variables before invoking ``2.py``.
"""

# Roostoo API credentials
ROOSTOO_API_KEY = "R9w8LnUa9O9cXjdggFlhP2mbblSiCSoxfPnLE6VUR1mQZUgsqgMYiuNyHUU1pwK5"
ROOSTOO_SECRET_KEY = "cJASSuT7pMmyc2Cx9vYMOl0nO8VnxlaxYFxJjmDKMs7ZqEpfcI09wyIhTJbpHce1"

# Horus market data API credential
HORUS_API_KEY = "78732c7f065ebee7e63c0b313628cc3a95e0e805ae6e237f59e445c69e3a1d8d"

# Optional: default poll interval override (seconds)
# Leave as ``None`` to use the value hardcoded in ``2.py``
POLL_INTERVAL_SECONDS = None
