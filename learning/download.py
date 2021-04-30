from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import os, time, sys

# flickr api
key = ""
secret = ""
wait_time = 1

# 保存先の指定
color_name = sys.argv[1]
savedir = "./img/" + color_name

# flickrAPI
flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = color_name,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence'
)

photos = result['photos']

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    # 同名ファイルがなければ次の処理へ
    if os.path.exists(filepath): continue
    # 同名ファイルがなければ画像DL
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)