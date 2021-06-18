import unicodedata
import re
from urllib.parse import unquote

def normalize(text):
    """Resolve different type of unicode encodings / capitarization in HotpotQA data."""
    text = unicodedata.normalize('NFD', text)
    return text[0].capitalize() + text[1:]

def find_hyper_linked_titles(text_w_links):
    titles = re.findall(r'href=[\'"]?([^\'" >]+)', text_w_links)
    titles = [unquote(title) for title in titles]
    titles = [title[0].capitalize() + title[1:] for title in titles]
    return titles

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)
