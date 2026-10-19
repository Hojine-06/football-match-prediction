import time
import requests
from config import RAW_DIR, LEAGUES, SEASONS, BASE_URL

HEADERS = {"User-Agent": "Mozilla/5.0 (football-match-prediction student project)"}

def fetch(url, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(2 * (attempt + 1))
    print(f"   erreur réseau : {last}")
    return None

def download_all():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok, missing = 0, []
    for season in SEASONS:
        for code in LEAGUES:
            dest = RAW_DIR / f"{code}_{season}.csv"
            if dest.exists():
                ok += 1
                continue
            url = BASE_URL.format(season=season, league=code)
            r = fetch(url)
            if r is not None and r.status_code == 200 and len(r.content) > 500:
                dest.write_bytes(r.content)
                ok += 1
                print(f"OK       {dest.name}")
            else:
                status = r.status_code if r is not None else "réseau"
                missing.append(f"{code}_{season} ({status})")
                print(f"MANQUANT {code} {season} ({status})")
            time.sleep(0.5)
    print(f"\nTerminé : {ok} fichiers présents, {len(missing)} manquants")
    for m in missing:
        print("  -", m)

if __name__ == "__main__":
    download_all()