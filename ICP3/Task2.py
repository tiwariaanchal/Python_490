import requests
from bs4 import BeautifulSoup
import os
html = requests.get("https://en.wikipedia.org/wiki/Deep_learning")
bsObj = BeautifulSoup(html.content, "html.parser")
print(bsObj.title.string)
for link in bsObj.find_all('a'):
    print(link.get('href'))



