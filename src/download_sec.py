from sec_edgar_downloader import Downloader
from pathlib import Path

SEC_DIR = Path("data/sec_filings")

# Financial + insurance companies — chosen for relevance to Manulife/finance roles
TICKERS = [
    "JPM",   # JPMorgan Chase
    "MET",   # MetLife
    "PRU",   # Prudential Financial
    "GS",    # Goldman Sachs
    "AIG",   # AIG (insurance)
]

FILINGS_PER_COMPANY = 2


def download():
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader("RAGDemo", "demo@ragdemo.com", SEC_DIR)

    for ticker in TICKERS:
        print(f"Downloading 10-K for {ticker}...")
        try:
            dl.get("10-K", ticker, limit=FILINGS_PER_COMPANY)
            print(f"  Done: {ticker}")
        except Exception as e:
            print(f"  Failed {ticker}: {e}")


if __name__ == "__main__":
    download()
