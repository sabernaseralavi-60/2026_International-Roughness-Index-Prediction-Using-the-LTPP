"""
Download every LTPP Standard Data Release (SDR 39) state/province ZIP from the
public InfoPave CloudFront distribution, extract the Primary Data Access
database, run 01_extract_state.py against it, then discard the large
Access/ZIP files (only the extracted per-state CSVs are kept under
data/raw/by_state/, since they are what stays in the git repo).

Source: https://infopave.fhwa.dot.gov/Data/StandardDataRelease/
(public-domain FHWA data; ZIPs are hosted at
https://du993ylnpbddg.cloudfront.net/SDR/39/By_State_Province/SDR39_<AB>.ZIP
and require no login)
"""
import os
import shutil
import subprocess
import sys
import zipfile
import urllib.request

STATES = {
    851: ("AK", "Alaska"), 852: ("AL", "Alabama"), 853: ("AR", "Arkansas"),
    854: ("AZ", "Arizona"), 855: ("BC", "British Columbia"), 856: ("CA", "California"),
    857: ("CO", "Colorado"), 858: ("CT", "Connecticut"), 859: ("DC", "District of Columbia"),
    860: ("DE", "Delaware"), 861: ("FL", "Florida"), 862: ("GA", "Georgia"),
    863: ("HI", "Hawaii"), 864: ("IA", "Iowa"), 865: ("ID", "Idaho"),
    866: ("IL", "Illinois"), 867: ("IN", "Indiana"), 868: ("KS", "Kansas"),
    869: ("KY", "Kentucky"), 870: ("LA", "Louisiana"), 871: ("MA", "Massachusetts"),
    872: ("MB", "Manitoba"), 873: ("MD", "Maryland"), 874: ("ME", "Maine"),
    875: ("MI", "Michigan"), 876: ("MN", "Minnesota"), 877: ("MO", "Missouri"),
    878: ("MS", "Mississippi"), 879: ("OH", "Ohio"), 880: ("OK", "Oklahoma"),
    881: ("ON", "Ontario"), 882: ("OR", "Oregon"), 883: ("PA", "Pennsylvania"),
    884: ("PE", "Prince Edward Island"), 885: ("PR", "Puerto Rico"), 886: ("QC", "Quebec"),
    887: ("RI", "Rhode Island"), 888: ("SC", "South Carolina"), 889: ("SD", "South Dakota"),
    890: ("SK", "Saskatchewan"), 891: ("TN", "Tennessee"), 892: ("TX", "Texas"),
    893: ("UT", "Utah"), 894: ("VA", "Virginia"), 895: ("VT", "Vermont"),
    896: ("WA", "Washington"), 897: ("WI", "Wisconsin"), 898: ("WV", "West Virginia"),
    899: ("WY", "Wyoming"), 900: ("AB", "Alberta"), 901: ("MT", "Montana"),
    902: ("NB", "New Brunswick"), 903: ("NC", "North Carolina"), 904: ("ND", "North Dakota"),
    905: ("NE", "Nebraska"), 906: ("NL", "Newfoundland"), 907: ("NH", "New Hampshire"),
    908: ("NJ", "New Jersey"), 909: ("NM", "New Mexico"), 910: ("NS", "Nova Scotia"),
    911: ("NV", "Nevada"), 912: ("NY", "New York"),
}

BASE_URL = "https://du993ylnpbddg.cloudfront.net/SDR/39/By_State_Province/SDR39_{}.ZIP"
HEADERS = {"User-Agent": "Mozilla/5.0"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw", "by_state")
TMP_DIR = os.path.join(ROOT, "data", "_tmp_sdr")


def run_one(sdr_id, abbr, name):
    out_flex = os.path.join(RAW_DIR, f"flexible_{abbr}.csv")
    out_rigid = os.path.join(RAW_DIR, f"rigid_{abbr}.csv")
    if os.path.exists(out_flex) or os.path.exists(out_rigid):
        print(f"SKIP {abbr} ({name}) - already extracted")
        return
    os.makedirs(TMP_DIR, exist_ok=True)
    zip_path = os.path.join(TMP_DIR, f"SDR39_{abbr}.ZIP")
    url = BASE_URL.format(abbr)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:
        print(f"FAIL download {abbr}: {e}")
        return

    extract_dir = os.path.join(TMP_DIR, abbr)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    except Exception as e:
        print(f"FAIL unzip {abbr}: {e}")
        return

    accdb = None
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            if fn.lower().endswith("_primary_data.accdb"):
                accdb = os.path.join(root, fn)
    if not accdb:
        print(f"FAIL no accdb found for {abbr}")
    else:
        try:
            subprocess.run(
                [sys.executable, os.path.join(ROOT, "code", "01_extract_state.py"),
                 accdb, abbr, RAW_DIR],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"FAIL extract {abbr}: {e}")

    shutil.rmtree(extract_dir, ignore_errors=True)
    os.remove(zip_path)


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    for sdr_id, (abbr, name) in STATES.items():
        run_one(sdr_id, abbr, name)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("DONE")
