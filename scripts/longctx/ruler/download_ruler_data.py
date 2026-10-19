"""Download the raw data RULER's generators need, the same way upstream RULER does.

    python scripts/longctx/ruler/download_ruler_data.py --out_dir ~/datasets/ruler/raw

- ``PaulGrahamEssays.json``: essay haystack. 49 essays from gkamradt/LLMTest_NeedleInAHaystack + 168 scraped from
  paulgraham.com (``<font>`` tag -> html2text), concatenated in sorted-filename order, as in RULER's
  ``scripts/data/synthetic/json/download_paulgraham_essay.py``. Needs ``html2text`` and ``beautifulsoup4``.
- ``squad.json`` (SQuAD v2.0 dev) and ``hotpotqa.json`` (HotpotQA dev distractor) for qa_1 / qa_2.

The URL list is RULER's ``PaulGrahamEssays_URLs.txt`` (pass ``--urls``, or it is fetched from GitHub).
"""

import argparse
import json
import os
import urllib.request


RULER_RAW = "https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic/json"
QA_URLS = {
    "squad.json": ["https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"],
    "hotpotqa.json": [
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/"
        "hotpot_dev_distractor_v1.json",
    ],
}


def fetch(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def download_essays(urls, out_path: str) -> None:
    import html2text
    from bs4 import BeautifulSoup

    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.ignore_tables = True
    converter.escape_all = True
    converter.reference_links = False
    converter.mark_code = False

    repo_texts, html_texts = {}, {}
    for url in urls:
        try:
            if ".html" in url:
                content = fetch(url).decode("unicode_escape", "utf-8")
                tag = BeautifulSoup(content, "html.parser").find("font")
                html_texts[url.split("/")[-1].replace(".html", ".txt")] = converter.handle(str(tag))
            else:
                repo_texts[url.split("/")[-1]] = fetch(url).decode("utf-8")
        except Exception as e:  # noqa: BLE001 - upstream also skips failed essays
            print(f"failed {url}: {e}")
    print(f"downloaded {len(repo_texts)} repo essays + {len(html_texts)} paulgraham.com essays (of {len(urls)})")
    text = "".join(repo_texts[k] for k in sorted(repo_texts)) + "".join(html_texts[k] for k in sorted(html_texts))
    with open(out_path, "w") as f:
        json.dump({"text": text, "num_essays": len(repo_texts) + len(html_texts)}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--urls", default=None, help="Local copy of RULER's PaulGrahamEssays_URLs.txt.")
    parser.add_argument("--skip_qa", action="store_true")
    args = parser.parse_args()
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    essay_path = os.path.join(out_dir, "PaulGrahamEssays.json")
    if not os.path.exists(essay_path):
        if args.urls:
            with open(os.path.expanduser(args.urls)) as f:
                urls = f.read().split()
        else:
            urls = fetch(f"{RULER_RAW}/PaulGrahamEssays_URLs.txt").decode().split()
        download_essays(urls, essay_path)

    if not args.skip_qa:
        for name, candidates in QA_URLS.items():
            path = os.path.join(out_dir, name)
            if os.path.exists(path):
                continue
            for url in candidates:
                try:
                    data = fetch(url, timeout=120)
                    json.loads(data)
                    with open(path, "wb") as f:
                        f.write(data)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"failed {url}: {e}")


if __name__ == "__main__":
    main()
