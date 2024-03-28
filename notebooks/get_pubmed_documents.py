from tqdm import tqdm
import time
from bs4 import BeautifulSoup
import requests
import json


def get_title_abstract(url) -> str:
    """
    Obtain PubMed paper abstract by its url
    """

    time.sleep(0.1)

    r = requests.get(url)
    bs = BeautifulSoup(r.text, "html.parser")
    # can be extended with other parts
    abstract = bs.find_all("div", {"abstract-content selected"})  # get abstract
    title = bs.find_all("h1", {"heading-title"})  # get title
    # similar_articles = [f"http://www.ncbi.nlm.nih.gov/pubmed/{x['data-ga-action']}" for x in bs.find_all(
    #    "div", {"similar-articles"})[0].find_all("a", {"docsum-title"})] # get the urls to similar articles
    # date = bs.find('meta', attrs={'name': 'citation_date'})['content'] # get the publication date

    if not abstract:
        abstract = ""
    else:
        abstract = abstract[0].get_text(strip=True)
    if not title:
        title = ""
    else:
        title = title[0].get_text(strip=True)
    return title, abstract
