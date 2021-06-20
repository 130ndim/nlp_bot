import json

import os
import os.path as osp

import string
import ssl

from typing import List

import urllib.request
import urllib.error
from urllib.parse import urljoin

import bs4

from tqdm.auto import tqdm


ssl._create_default_https_context = ssl._create_unverified_context


class BeerScraper:
    __base_url__ = 'https://spb.winestyle.ru'

    __product_attrs_ = {'Пиво', 'Стиль', 'Регион', 'Производитель', 'Бренд', 'Тип ферментации', 'Крепость', 'Объем'}

    def __init__(self, path, category='beer'):
        self.path = path
        self.img_path = osp.join(path, 'images')
        self.category = category

        os.makedirs(self.img_path, exist_ok=True)

    def parse(self, max_idx=None):
        bar = tqdm()
        idx = 0
        data = {}
        while True:
            idx += 1
            try:
                for product in self.parse_single_page(idx):
                    try:
                        parsed_product = self.parse_product(product)
                        data[osp.basename(product)[:-5]] = parsed_product
                    except Exception as e:
                        print(e)
                        print(product)
                    bar.update(1)
            except urllib.error.HTTPError:
                break
            if max_idx is not None and idx == max_idx:
                break
        with open(osp.join(self.path, 'data.json'), 'w') as f:
            json.dump(data, f)
        return data

    def parse_product(self, product: str):
        if product.startswith('/'):
            product = product[1:]
        url = urljoin(self.__base_url__, product)
        request = urllib.request.Request(url)
        html = urllib.request.urlopen(request).read()
        soup = bs4.BeautifulSoup(html, 'html.parser')
        list_desc = soup.find('ul', 'list-description list-description-lined')
        data = {}
        for el in list_desc.find_all('li', ''):
            # print(el.get_text())
            parsed = el.get_text().replace(':', '')
            parsed = parsed.replace(' / ', '/')
            if len(el) == 0:
                continue
            _, field, attrs = parsed.split('\n', maxsplit=2)
            if field in self.__product_attrs_:
                data[field] = ' '.join(attrs.split()).split(', ')

        img_url = soup.find('a', 'img-container fancybox')['href']
        product_name = osp.basename(product)
        save_path = osp.join(self.img_path, product_name[:-5] + '.' + img_url.split('.')[-1])
        urllib.request.urlretrieve(img_url, save_path)
        data['img'] = save_path
        data['price'] = soup.find('div', 'price').text

        temp = soup.find(text='Температура сервировки:')
        if temp is not None:
            data['temperature'] = temp.parent.parent.parent.text.split()[-2]

        ingredients = soup.find(text='Состав:')
        if ingredients is not None:
            data['ingredients'] = ingredients.parent.parent.parent.text.replace(',', '').split()[1:]

        tags = soup.find_all('div', 'tag-block')
        tastes = tags[0].find_all('a')
        if tastes is not None:
            data['tastes'] = [a.text for a in tastes]
        serve_with = tags[1].find_all('a')
        if serve_with is not None:
            data['serve_with'] = [a.text for a in serve_with]
        return data

    def parse_single_page(self, idx: int) -> List[str]:
        url = urljoin(self.__base_url__, f'beer/all/?page={idx}')
        request = urllib.request.Request(url)
        html = urllib.request.urlopen(request).read()

        soup = bs4.BeautifulSoup(html, 'html.parser')
        hrefs = [p.a['href'] for p in soup.find_all('p', 'title')]
        return hrefs


if __name__ == '__main__':
    scraper = BeerScraper('../data')
    req = urllib.request.Request('https://spb.winestyle.ru/products/Lindemans-Gueuze.html')
    html = urllib.request.urlopen(req).read()
    soup = bs4.BeautifulSoup(html, 'html.parser')

    # print(soup.prettify())
    # print(soup.find(text='Состав:').parent.parent.parent.text.replace(',', '').split())
    scraper.parse()
    # s = soup.find_all('div', 'tag-block')
    # print(s[0].find_all('a'))
    # print(s.translate(str.maketrans('', '', string.punctuation)).split())
    # print(soup.find_all('div', 'description-block'))
    # print(scraper.parse_product('products/Lindemans-Gueuze.html'))
