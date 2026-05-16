"""
scraper.py — Updated for Pydantic v2.
Key change: replaced @validator (v1) with @field_validator (v2).
All scraping logic is unchanged from original.
"""

import time
import json
import os
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl, field_validator
import pandas as pd


# ---------------------------------------------------------------------------
# Pydantic v2 model
# ---------------------------------------------------------------------------

class CatalogItem(BaseModel):
    name: str
    url: HttpUrl
    duration: str
    test_type: str
    remote_testing: str
    adaptive_irt: str
    tags: List[str]

    @field_validator("test_type")
    @classmethod
    def validate_test_type_keys(cls, v: str) -> str:
        allowed = {"A", "K", "P", "B", "C", "S", "N/A"}
        keys = [k.strip() for k in v.split(",") if k.strip()]
        for k in keys:
            if k not in allowed:
                print(f"⚠️  Warning: unknown test type key '{k}' in '{v}'")
        return v


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class SHLCatalogScraper:
    BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }

    def scrape_individual_tests(self) -> List[Dict[str, Any]]:
        all_assessments: List[Dict[str, Any]] = []
        start = 0
        page_size = 12
        page_num = 1

        while True:
            url = f"{self.BASE_URL}?start={start}&type=1"
            print(f"📄 Fetching page {page_num}: {url}")
            resp = requests.get(url, headers=self.HEADERS, timeout=15)

            if resp.status_code != 200:
                print(f"❌ HTTP {resp.status_code} — stopping.")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                print("⚠️  No table found — stopping.")
                break

            page_items = self._parse_table(table)
            if not page_items:
                print("⚠️  No rows parsed — stopping.")
                break

            all_assessments.extend(page_items)
            print(f"   ✅ {len(page_items)} items (total: {len(all_assessments)})")

            # Pagination
            next_link = soup.find("a", string="Next") or soup.find("a", class_="next")
            if not next_link:
                print("🏁 No 'Next' link — last page.")
                break
            if (
                next_link.get("aria-disabled") == "true"
                or next_link.find_parent("li", class_="disabled")
            ):
                print("🏁 'Next' disabled — last page.")
                break
            if len(page_items) < page_size:
                print(f"🏁 Partial page ({len(page_items)} items) — last page.")
                break

            start += page_size
            page_num += 1
            time.sleep(0.5)

        if len(all_assessments) < 100:
            print(f"⚠️  Only {len(all_assessments)} items — check the site HTML.")
        else:
            print(f"✅ Total scraped: {len(all_assessments)}")
        return all_assessments

    def _parse_table(self, table) -> List[Dict[str, Any]]:
        rows = table.find_all("tr")[1:]
        items = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            link = cols[0].find("a")
            if not link:
                continue
            name = link.get_text(strip=True)
            href = link.get("href", "")
            full_url = f"https://www.shl.com{href}" if href else ""

            remote_testing = "Yes" if cols[1].find("span", class_="catalogue__circle -yes") else "No"
            adaptive_irt   = "Yes" if cols[2].find("span", class_="catalogue__circle -yes") else "No"
            key_spans = cols[3].find_all("span", class_="product-catalogue__key")
            test_type = ", ".join(s.get_text(strip=True) for s in key_spans) or "N/A"

            items.append({
                "name": name,
                "url": full_url,
                "duration": "N/A",
                "test_type": test_type,
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
                "tags": self._infer_tags(name, test_type),
            })
        return items

    def _infer_tags(self, name: str, test_type: str) -> List[str]:
        tags = set()
        n = name.lower()
        t = test_type.upper()

        keyword_map = {
            "technical": ["java", "python", "coding", "programming", ".net", "c++",
                          "javascript", "sql", "scala", "r ", "spark", "hadoop"],
            "management": ["manager", "leadership", "executive", "director"],
            "sales":      ["sales"],
            "customer-service": ["customer service", "customer care"],
            "cognitive":  ["cognitive", "ability", "aptitude", "numerical", "verbal",
                           "inductive", "deductive"],
            "personality": ["personality", "opq"],
            "behavioral":  ["behavioral", "situational"],
            "graduate":    ["graduate", "entry level", "entry-level"],
        }
        for tag, keywords in keyword_map.items():
            if any(kw in n for kw in keywords):
                tags.add(tag)

        type_code_map = {"A": "ability", "P": "personality", "K": "knowledge",
                         "B": "behavioral", "C": "competency", "S": "situational"}
        for code, tag in type_code_map.items():
            if code in t:
                tags.add(tag)

        return list(tags) if tags else ["general"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🚀 Phase 1: Scraping SHL Individual Test Solutions…")
    scraper = SHLCatalogScraper()
    raw = scraper.scrape_individual_tests()
    print(f"\n📊 Raw items: {len(raw)}")

    valid, dropped = [], 0
    for item in raw:
        try:
            CatalogItem(**item)
            # Pydantic v2: url is a Url object — convert back to str for JSON
            item["url"] = str(item["url"])
            valid.append(item)
        except Exception as e:
            print(f"❌ Dropped '{item.get('name')}': {e}")
            dropped += 1

    print(f"✅ Valid: {len(valid)}, dropped: {dropped}")

    os.makedirs("data", exist_ok=True)
    with open("data/catalog.json", "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)
    print("💾 Saved data/catalog.json")

    df = pd.DataFrame(valid)
    df.to_csv("data/shl_catalog.csv", index=False)
    print("💾 Saved data/shl_catalog.csv")
    print("✅ Phase 1 complete.")