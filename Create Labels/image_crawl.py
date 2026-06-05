import requests, os
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

url = "https://www.iplt20.com/photos/2486"
res = requests.get(url)
soup = BeautifulSoup(res.text, "html.parser")

os.makedirs("IPL_2486", exist_ok=True)

def download(src):
    filename = os.path.join("IPL_2486", src.split("/")[-1])
    if not os.path.exists(filename):
        try:
            r = requests.get(src, timeout=10)
            with open(filename, "wb") as f:
                f.write(r.content)
            print("Downloaded:", filename)
        except Exception as e:
            print("Failed:", src, e)

sources = []
for div in soup.find_all("div", class_="ap-photo-inner-wrp"):
    img = div.find("img")
    if img:
        src = img.get("src") or img.get("data-src")
        if src and src.startswith("http"):
            sources.append(src)

with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(download, sources)