import argparse
import json
import os
import re
import time
from collections import deque
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


URDU_PUNCT = {"۔", "؟", "!", "،", "؛", ":", ".", "?", ","}


def extract_article_fields(soup):
    title = ""
    publish_date = ""

    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and next_data.string:
        try:
            data = json.loads(next_data.string)
            page_data = data["props"]["pageProps"]["pageData"]
            analytics = page_data["metadata"]["atiAnalytics"]
            title = analytics.get("pageTitle", "")
            publish_date = analytics.get("timePublished", "")
        except Exception:
            pass

    main = soup.find("main", role="main") or soup
    paragraphs = main.find_all("p", dir="rtl")
    body_text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    return title.strip() or "No Title", publish_date.strip() or "Unknown Date", body_text.strip()


def save_raw_txt(raw_articles, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for a in raw_articles:
            f.write(f"[{a['article_number']}]\n{a['body']}\n\n")


def save_metadata_json(metadata_records, output_path):
    obj = {
        str(r["article_number"]): {
            "title": r["title"],
            "publish_date": r["publish_date"],
            "source_url": r["source_url"],
        }
        for r in metadata_records
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def scrape_bbc_urdu(target_count=260, min_body_chars=300, sleep_s=0.3):
    base_url = "https://www.bbc.com"
    start_pages = [
        "https://www.bbc.com/urdu",
        "https://www.bbc.com/urdu/topics/cjgn7n9zzq7t",
        "https://www.bbc.com/urdu/topics/cl8l9mveql2t",
        "https://www.bbc.com/urdu/topics/cw57v2pmll9t",
        "https://www.bbc.com/urdu/topics/c340q0p2585t",
        "https://www.bbc.com/urdu/topics/ckdxnx900n5t",
        "https://www.bbc.com/urdu/topics/c40379e2ymxt",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    session = requests.Session()
    queue = deque(start_pages)
    visited = set()

    raw_articles = []
    metadata_records = []

    print(f"Crawling BBC Urdu for up to {target_count} articles...")
    while queue and len(raw_articles) < target_count:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            response = session.get(url, headers=headers, timeout=15)
        except Exception:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/urdu/articles/" in href:
                full = urljoin(base_url, href)
                clean = full.split("?")[0].split("#")[0]
                if clean not in visited:
                    queue.append(clean)

        if "/urdu/articles/" in url:
            title, publish_date, body = extract_article_fields(soup)
            if len(body) >= min_body_chars:
                number = len(raw_articles) + 1
                raw_articles.append(
                    {
                        "article_number": number,
                        "body": body,
                    }
                )
                metadata_records.append(
                    {
                        "article_number": number,
                        "title": title,
                        "publish_date": publish_date,
                        "source_url": url,
                    }
                )
                if number == 1 or number % 10 == 0:
                    print(f"  scraped {number}/{target_count}")

        time.sleep(sleep_s)

    return raw_articles, metadata_records


def read_raw_blocks(raw_path):
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n(?=\[\d+\])", text) if b.strip()]
    parsed = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        match = re.fullmatch(r"\[(\d+)\]", lines[0].strip())
        if not match:
            continue
        article_no = int(match.group(1))
        body = "\n".join(lines[1:]).strip()
        parsed.append({"article_number": article_no, "body": body})
    return parsed


def remove_diacritics(text):
    return re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)


def normalize_urdu_unicode(text):
    char_map = {
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "ة": "ہ",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "آ": "ا",
        "ؤ": "و",
        "ۀ": "ہ",
        "ہٰ": "ہ",
    }
    for src, dst in char_map.items():
        text = text.replace(src, dst)
    return text


def remove_noise(text):
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[A-Za-z]+", " ", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]+", " ", text)
    text = re.sub(r"[^\u0600-\u06FF0-9۰-۹\s\.,!?؟،؛:۔\"'\-()\[\]/]", " ", text)
    return text


def normalize_whitespace(text):
    text = re.sub(r"\s*([۔!?؟،؛:])\s*", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def segment_sentences_urdu(text):
    text = re.sub(r"\.(?=\s|$)", "۔", text)
    parts = re.split(r"(?<=[۔!?؟])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def clean_article_text(body_text):
    text = remove_diacritics(body_text)
    text = normalize_urdu_unicode(text)
    text = remove_noise(text)
    text = normalize_whitespace(text)
    if not text:
        return []
    return segment_sentences_urdu(text)


def replace_numbers(text):
    return re.sub(r"[0-9۰-۹]+", " <NUM> ", text)


def custom_urdu_tokenize(sentence):
    sentence = replace_numbers(sentence)
    sentence = re.sub(r"([۔؟!،؛:\.,\?\(\)\[\]\-])", r" \1 ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    tokens = []
    for token in sentence.split(" "):
        token = token.strip("'\"“”‘’`")
        if token:
            tokens.append(token)
    return tokens


COMMON_URDU_SUFFIXES = sorted(
    {
        "اتیں",
        "گان",
        "یاں",
        "یوں",
        "وں",
        "یں",
        "گی",
        "گا",
        "نے",
        "نا",
        "تا",
        "تی",
        "تے",
        "دی",
        "دا",
        "ے",
        "ی",
    },
    key=len,
    reverse=True,
)


def custom_urdu_stem(token):
    if token == "<NUM>" or token in URDU_PUNCT or len(token) <= 2:
        return token
    for suffix in COMMON_URDU_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)]
    return token


PLURAL_LEMMA_MAP = {
    "لڑکیاں": "لڑکی",
    "کتابوں": "کتاب",
    "کتابیں": "کتاب",
    "بچوں": "بچہ",
    "لوگوں": "لوگ",
}

GENDER_LEMMA_MAP = {
    "اچھی": "اچھا",
    "بڑی": "بڑا",
    "چھوٹی": "چھوٹا",
    "نئی": "نیا",
}


def custom_urdu_lemmatize(token):
    if token == "<NUM>" or token in URDU_PUNCT:
        return token
    if token in PLURAL_LEMMA_MAP:
        return PLURAL_LEMMA_MAP[token]
    if token in GENDER_LEMMA_MAP:
        return GENDER_LEMMA_MAP[token]
    if len(token) > 3 and token.endswith("یاں"):
        return token[:-3] + "ی"
    if len(token) > 3 and token.endswith("وں"):
        return token[:-2]
    if len(token) > 3 and token.endswith("یں"):
        return token[:-2]
    return token


def write_processed_txt(path, articles):
    with open(path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(f"[{article['article_number']}]\n")
            for sentence in article["lemmatized_sentences"]:
                tokens = [t for t in sentence if t not in URDU_PUNCT]
                if tokens:
                    f.write(" ".join(tokens) + "\n")
            f.write("\n")


def process_to_cleaned(raw_path, cleaned_path):
    raw_articles = read_raw_blocks(raw_path)
    processed = []
    for article in raw_articles:
        sentences = clean_article_text(article["body"])
        tokenized = [custom_urdu_tokenize(s) for s in sentences if s.strip()]
        stemmed = [[custom_urdu_stem(t) for t in sent] for sent in tokenized]
        lemmatized = [[custom_urdu_lemmatize(t) for t in sent] for sent in stemmed]
        processed.append(
            {
                "article_number": article["article_number"],
                "lemmatized_sentences": lemmatized,
            }
        )
    write_processed_txt(cleaned_path, processed)
    return len(processed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_count", type=int, default=260)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--min_body_chars", type=int, default=300)
    args = parser.parse_args()

    for fp in ["raw.txt", "cleaned.txt", "metadata.json", "Metadata.json"]:
        if os.path.exists(fp):
            os.remove(fp)

    raw_articles, metadata_records = scrape_bbc_urdu(
        target_count=args.target_count,
        min_body_chars=args.min_body_chars,
        sleep_s=args.sleep,
    )

    save_raw_txt(raw_articles, "raw.txt")
    save_metadata_json(metadata_records, "metadata.json")

    with open("metadata.json", "r", encoding="utf-8") as source:
        metadata_obj = json.load(source)
    with open("Metadata.json", "w", encoding="utf-8") as target:
        json.dump(metadata_obj, target, ensure_ascii=False, indent=2)

    cleaned_count = process_to_cleaned("raw.txt", "cleaned.txt")

    print(f"Saved raw.txt with {len(raw_articles)} articles")
    print(f"Saved metadata.json and Metadata.json with {len(metadata_records)} records")
    print(f"Saved cleaned.txt with {cleaned_count} articles")


if __name__ == "__main__":
    main()
