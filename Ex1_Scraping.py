import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time

def download_images(search_query, folder_name, num_images=1100):
    driver = webdriver.Chrome() 
    driver.get(f"https://yandex.ru/images/search?text={search_query}")

    time.sleep(5)

    images = set()
    while len(images) < num_images:
        soup = BeautifulSoup(driver.page_source, 'lxml')
        img_tags = soup.find_all('img')
        print(len(img_tags))
        for img in img_tags:
            img_url = img.get('src')
            if img_url and img_url not in images:
                images.add(img_url)
                if len(images) >= num_images:
                    break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2) 

    driver.quit()
    
    for i, img_url in enumerate(images):
        if i >= num_images:
            break
        response = requests.get('https:'+img_url)
        file_name = f"{folder_name}/{str(i).zfill(4)}.jpg"
        with open(file_name, 'wb') as f:
            f.write(response.content)

download_images("polar bear", "dataset/polar_bear")
download_images("brown bear", "dataset/brown_bear")