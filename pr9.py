import requests
import bs4

url = "https://www.flipkart.com/poco-c31-royal-blue-64-gb-/p/itm19effae969b86"

headers = {
    "User-Agent": "Mozilla/5.0"
}

request1 = requests.get(url, headers=headers)

print("STATUS CODE =", request1.status_code)

soup = bs4.BeautifulSoup(request1.text, "html.parser")

print(soup.title.get_text())