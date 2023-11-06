"""
# 셀레니움_기본설정
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# 크롬 드라이버 자동 업데이트
from webdriver_manager.chrome import ChromeDriverManager

# 브라우저 꺼짐 방지
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

# 불필요한 에러 메시지 없애기
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

service = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# 웹페이지 주소 이동
driver.get("http://www.naver.com")

# 3. 네이버 로그인 자동화

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# 크롬 드라이버 자동 업데이트
from webdriver_manager.chrome import ChromeDriverManager

import time
import pyautogui
import pyperclip

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

# 불필요한 에러 메시지 없애기
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

service = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# 웹페이지 주소 이동
driver.implicitly_wait(5)  # 웹페이지가 로딩 될때까지 5초는 기다림
driver.maximize_window()  # 화면 최대화
driver.get("https://nid.naver.com/nidlogin.login?mode=form&url=https://www.naver.com/")

# 아이디 입력창
id = driver.find_element(By.CSS_SELECTOR, "#id")
id.click()
# id.send_keys("tldus0_0")
pyperclip.copy("tldus0_0")
pyautogui.hotkey("ctrl", "v")
time.sleep(2)

# 비밀번호 입력창
pw = driver.find_element(By.CSS_SELECTOR, "#pw")
pw.click()
# pw.send_keys("   ")
pyperclip.copy("   ")
pyautogui.hotkey("ctrl", "v")
time.sleep(2)

# 로그인 버튼
login_btn = driver.find_element(By.CSS_SELECTOR, "#log\.login")
login_btn.click()
""" """
import requests
from bs4 import BeautifulSoup

# naver 서버에 대화를 시도
response = requests.get("https://finance.naver.com/")

# naver에서 html 줌
html = response.text

# html 번역선성님으로 수프를 만듦
soup = BeautifulSoup(html, "html.parser")

# id 값이 NM_set_home_btn인 놈 한개를 찾아냄
Word = soup.select_one("#menu > ul > li.m1.first.on > a > span.tx")

# 텍스트 요소만 출력
print(Word)
"""
import requests
from bs4 import BeautifulSoup

response = requests.get(
    "https://search.naver.com/search.naver?where=news&sm=tab_jum&query=%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90"
)
html = response.text
soup = BeautifulSoup(html, "html.parser")
links = soup.select(".news_tit")  # 결과는 리스트

for link in links:
    title = link.text  # 태그 안에 텍스트요소를 가져온다
    url = link.attrs["href"]  # href의 속성값을 가져온다
    print(title, url)
