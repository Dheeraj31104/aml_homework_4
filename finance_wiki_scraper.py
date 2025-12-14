#!/usr/bin/env python3
"""
Finance Multi-Site Scraper - Scrapes finance topics from Wikipedia and other finance sites
Run: python finance_wiki_scraper.py
"""

import json
from datetime import datetime
from typing import List

import requests
from bs4 import BeautifulSoup


class FinanceWikiScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        self.documents = []

    def _scrape_page(self, url: str, fallback_title: str):
        response = requests.get(url, headers=self.headers, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"status {response.status_code}")

        soup = BeautifulSoup(response.content, "html.parser")
        title_tag = soup.find("h1") or soup.find("title")
        title_text = title_tag.get_text().strip() if title_tag else fallback_title

        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text() for p in paragraphs[:8])
        content = " ".join(content.split())
        return title_text, content

    def scrape_finance_topics(self, topics: List[str]):
        """Scrape Wikipedia finance articles."""
        print("=" * 70)
        print("SCRAPING FINANCE WIKIPEDIA ARTICLES")
        print("=" * 70)

        for topic in topics:
            url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
            try:
                print(f"\n✓ Scraping: {topic}")
                title_text, content = self._scrape_page(url, topic)
                doc = {
                    "title": title_text,
                    "url": url,
                    "source": "Wikipedia",
                    "content": content,
                    "scraped_at": datetime.now().isoformat(),
                }
                self.documents.append(doc)
                print(f"  ✓ Success! {len(content)} chars")
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:80]}")

        return self.documents

    def scrape_finance_sites(self, urls: List[str]):
        """Scrape a list of finance sites (e.g., Google Finance, Yahoo Finance, Investopedia, MarketWatch)."""
        print("=" * 70)
        print("SCRAPING FINANCE NEWS/INFO SITES")
        print("=" * 70)

        for url in urls:
            try:
                print(f"\n✓ Scraping: {url}")
                title_text, content = self._scrape_page(url, url)
                doc = {
                    "title": title_text,
                    "url": url,
                    "source": "finance_site",
                    "content": content,
                    "scraped_at": datetime.now().isoformat(),
                }
                self.documents.append(doc)
                print(f"  ✓ Success! {len(content)} chars")
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:80]}")

        return self.documents

    def save(self, filename: str = "finance_knowledge_base.json"):
        """Save scraped documents to file."""
        with open(filename, "w") as f:
            json.dump(self.documents, f, indent=2)
        print(f"\n✓ Saved {len(self.documents)} documents to {filename}")


if __name__ == "__main__":
    scraper = FinanceWikiScraper()

    finance_topics = [
        "Stock",
        "Bond (finance)",
        "Government bond",
        "Corporate bond",
        "Fixed income",
        "Dividend",
        "Portfolio",
        "Investment",
        "Risk management",
    ]

    finance_sites = [
        "https://www.google.com/finance/markets",
        "https://finance.yahoo.com/",
        "https://www.investopedia.com/markets-news-4427782",
        "https://www.marketwatch.com/",
    ]

    scraper.scrape_finance_topics(finance_topics)
    scraper.scrape_finance_sites(finance_sites)
    scraper.save()
