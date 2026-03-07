"""Web crawler for collecting review data from forums and review sites."""

import csv
import random
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup


class ReviewCrawler:
    """Web crawler to collect review data from forums/review sites."""

    def __init__(self, base_url: str, output_file: str = 'data/raw/crawled_reviews.csv'):
        self.base_url = base_url
        self.output_file = output_file
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; SentimentBot/1.0; +research)'
        }
        self.reviews: list[dict] = []

    def crawl_page(self, url: str) -> list[dict]:
        """Fetch and parse a single page for review content."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            page_reviews = []
            # Adapt selectors to target site
            review_elements = soup.find_all('div', class_='review')
            for elem in review_elements:
                text = elem.get_text(strip=True)
                rating = elem.find('span', class_='rating')
                page_reviews.append({
                    'text': text,
                    'rating': rating.get_text(strip=True) if rating else None,
                    'source_url': url
                })

            return page_reviews

        except requests.RequestException as e:
            print(f"Error crawling {url}: {e}")
            return []

    def crawl_multiple_pages(self, urls: list[str], delay: float = 2.0):
        """Crawl multiple pages with polite delays."""
        for url in urls:
            reviews = self.crawl_page(url)
            self.reviews.extend(reviews)
            print(f"Collected {len(reviews)} reviews from {url}")
            time.sleep(delay + random.uniform(0, 1))

    def save_to_csv(self):
        """Save crawled reviews to CSV."""
        import os
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'rating', 'source_url'])
            writer.writeheader()
            writer.writerows(self.reviews)
        print(f"Saved {len(self.reviews)} reviews to {self.output_file}")
