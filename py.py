"""
print("hello world")
print(5)
print(-10)
print("3.12")
print(True)
print(False)
print(not True)
print(not (3-10))
""" """
animal = "고양이"
name = "고앵이"
age = 5
hobby = "낮잠"
print("우리집 "+animal+"의 이름은 "+name+"예요")
print(name+"는 "+str(age)+"살이며, "+hobby+"을 아주 좋아해요")
""" """
station = "사당"
print(station+"행 열차가 들어오고 있습니다.")
""" """
print(1+1)
print(3-2)
print(5*2)
print(6/3)

print(2**3)
print(5%3)
print(10//3)

print(10>3)
print(4>= 7)
print(10<3)
print(5<=5)

print(1 != 3)
print(not(1 != 3))

print((3>0) and (3<5))
print((3>0) & (3<5)) 
print((3>0) or (3<5)) 
print((3>0) | (3<5))

print(3<4<5)
print(5>3>7)
""" """
print(2+3*4)
print((3+2)*3)
num = 2 + 3* 4
print(num)
num = num + 2
print(num)
num+=2
print(num)
num %=3
print(num)
""" """
print(abs(-5)) # 절댓값
print(pow(4, 2)) # 4의 2승
print(max(5,12)) # 최댓값
print(min(5,12)) # 최솟값
print(round(3.14)) # 반올림
print(round(4.99)) # 반올림
from math import *
print(floor(4.99)) # 내림
print(ceil(3.14)) # 올림
print(sqrt(16)) # 제곱근
""" """
from random import *
print(random()) # 0.0 ~ 1.0 미만의 임의의 값 생성 
print(random()*10) # 0.0 ~ 10.0미만의 임의의 값 생성
print(int(random()*10)) # 0 ~ 10 미만의 임의의 값 생성
""" """
from random import *
date = randint(4,28)
print("오프라인 스터디 모임 날짜는 매월"+str(date)+"일로 선정되었습니다.")
""" '''
sentence1 = "나는 소년입니다."
print(sentence1)
sentence2 = "파이선은 어려워요"
print(sentence2)
sentence3 = """
나는 소년이고
파이션은 쉬워요
"""
print(sentence3)
''' """
jumin = "990120-1234567"
print("성별 : "+ jumin[7])
print("연 : " + jumin[0:2]) # 0부터 2직전까지(0~1)
print("월 : " + jumin[2:4])
print("일 : " + jumin[4:6])
print("생년월일 : "+ jumin[:6]) # 처음부터 6직전까지
print("뒤 7자리 : " + jumin[7:]) # 7부터 끝까지
print("뒤 7자리 (뒤에서부터) : " + jumin[-7:])# 맨 뒤에서 7번째부터 끝까지
""" """
python = "Python is Amazing"
print(python.lower()) # python is amazing
print(python.upper()) # PYTHON IS AMAZING
print(python[0].isupper()) # True
print(python[0].islower()) # False
print(len(python)) # 17
java = (python.replace("Python", "Java")) 
print(java) #Java is Amazing
index = python.index("n") 
print(index) # 5
print(python.index("n", index + 1)) # 15
print(python.find("n")) # 5
print(python.find("Java")) # -1
# print(python.index("Java")) #오류
print("hi") # hi
print(python.count("n")) # 2
print(python.count("Java")) # 0
""" """
print("나는 %d살입니다." % 20)
print("나는 %s을 좋아합니다." %"파이선")
print("Apple은 %c로 시작해요" %"A")
print("나는 %s살 입니다." %20)
print("나는 %s색과 %s색을 좋아해요." % ("파란", "빨간"))

print("나는 {}살 입니다.".format(20))
print("나는 {}색과 {}색을 좋아해요".format("파란", "빨간"))
print("나는 {1}색과 {0}색을 좋아해요.".format("파란", "빨강"))

print("나는 {age}살이며, {color}색을 좋아해요.".format(age = 20, color = "빨간"))

age = 20
color = "빨간"
print(f"나는 {age}살이며, {color}색을 좋아해요")
""" """
print("백문이 불여일견\n백견이 불여일타")

#저는 "류현성"입니다.
print('저는 "류현성"입니다.')
print("저는 \"류현성\"입니다.")
print("C:\\Users\\User>")

# \r : 커서를 맨 앞으로 이동
print("Red Apple\rPine")

# \b : 백스페이스 (한 글자 삭제)
print("Redd\bApple")

# \t : 탭
print("Red\t Apple")
""" """
# Quiz) 사이트별로 비밀번호를 만들어 주는 프로그램을 작성하시오

# 예) http://naver.com
# 규칙1 : http:// 부분은 제외 => naver.com
# 규칙2 : 처음 만나는 점(.) 이후 부분은 제외 => naver
# 규칙3 : 남은 글자 중 처음 세자리 + 글자 갯수 + 글자 내 'e' 갯수 + "!"로 구성
# 예) 생성된 비밀번호 : nav51!

url = "http://naver.com"
my_str = url.replace("http://", "")
print(my_str)
my_str = my_str[:my_str.index(".")] #my_str[0:5] -> 0 ~ 5 직전까지.(0.1.2.3.4)
print(my_str)
password = my_str[:3] + str(len(my_str)) + str(my_str.count("e")) + "!"
print(f"{url}의 비밀번호는 {password}입니다.")
""" """
# 리스트 []
subway = [10, 20, 30]
print(subway)

subway = ["유재석", "조세호", "박명수"]
print(subway)

# 조세호씨가 몇 번째 칸에 타고있는가?
print(subway.index("조세호"))

subway.append("하하")
print(subway)

# 정형돈씨를 유재석 / 조세호 사이에 태워봄
subway.insert(1, "정형돈")
print(subway)

print(subway.pop())
print(subway)

subway.append("유재석")
print(subway)
print(subway.count("유재석"))

num_list = [5,2,4,3,1]
num_list.sort()
print(num_list)
num_list.reverse()

print(num_list)

num_list.clear()
print(num_list)
""" """
cabinet = {3 : "류", 100 : "현"}
print(cabinet[3]) # 류
print(cabinet[100]) # 현

print(cabinet.get(3)) # 류

# print(cabinet[5]) # 캐비넷에 5가 없기 때문에 오류 후 종료
# print("hi") # 실행되지 않음

print(cabinet.get(5)) # none
print(cabinet.get(5, "성")) # 성

print(3 in cabinet) # True
print(5 in cabinet) # False

cabinet = {"A-3":"류", "B-100": "현"}
print(cabinet["A-3"]) # 류
print(cabinet["B-100"]) # 현

# 새 손님
print(cabinet)
cabinet["A-3"] = "nus"
cabinet["C-20"] = "Legi"
print(cabinet)

# 간 손님
del cabinet["A-3"]

# key 들만 출력
print(cabinet.keys())

# value 들만 출력
print(cabinet.values())

# key, value 쌍으로 출력
print(cabinet.items())

# 캐비넷 없애기
cabinet.clear()
print(cabinet)
""" """
menu = ("pork", "cheese")
print(menu[0]) # pork
print(menu[1]) # cheese

#menu.add("selmon") # error

name = "Ryu"
age = 20
hobby = "coding"
print(name, age, hobby)

name, age, hobby = "R", 2, "c"
print(name, age, hobby)
""" """
# 집합 (set)
# 중복 안됨, 순서 없음
my_set = {1,2,3,3,3,4}
print(my_set) # 중복 x

java = {"R", "Y", "U"}
python = set(["R", "L"])

# 교집합 ( java 와 python을 모두 할 수 있는 사람)
print(java & python)
print(java.intersection(python))

# 합집합 (java 할 수 있거나 python 할 수 있는 사람)
print(java | python)
print(java.union(python))

# 차집합(java 는 할 수 있지만 python 할 수 없는 사람)
print(java - python)
print(java.difference(python))

# python 할 줄 아는 사람이 늘어남
python.add("E")
print(python)

# java 까먹음
print(java)
java.remove("U")
print(java)
""" """
# 자료구조의 변경
# 커피숍
menu = {"커피", "우유", "주스"}
print(menu, type(menu))

menu = list(menu)
print(menu, type(menu))

menu = tuple(menu)
print(menu, type(menu))

menu = set(menu)
print(menu, type(menu))
""" """
# Quiz) 추첨을 통해 1명은 치킨, 3명은 커피 쿠폰을 받게 됩니다.

# 조건1 : 20명이 작성하였고 아이디는 1~20이라고 가정
# 조건2 : 댓글 내용과 상관 없이 무작위로 추첨하되 중복 불가
# 조건3 : random 모듈의 shuffle 과 sample 을 활용

# (출력 예제)
# -- 당첨자 발표 --
# 치킨 당첨자 : [1]
# 커피 당첨자 [2, 3, 5]
# -- 축하합니다 --

# (활용 예제)
# from random import *
# L = [1,2,3,4,5]
# print(L)
# shuffle(L)
# print(L)
# print(sample(L, 1))

from random import *
#L = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
L = range(1,21) # 1~20까지
L = list(L)
print(L)
shuffle(L)
print(L)

winners = sample(L,4) # 4명 중 1명은 치킨, 3명은 커피


print("-- 당첨자 발표 -- ")
print("치킨 당첨자 : [%d]" %(winners[0]))
print("커피 당첨자 : {0}".format(winners[1:]))
print("-- 축하합니다 --")
""" """
# weather = input("오늘 날씨는? ex)rain or dust\n")
# if weather == "rain" or weather == "snow":
#     print("우산을 챙기세요")
# elif weather == "dust":
#     print("마스크를 챙기세요")
# else:
#     print("준비물이 필요 없어요")
temp = int(input("기온은 어떄요\n"))if 30<= temp:
    print("너무 더워요. 나가지 마세요")
elif 10<= temp and temp < 30:
    print("괜찮은 날씨에요")
elif 0<= temp and temp < 10:
    print("외투를 챙기세요")
else:
    print("너무 추워요. 나가지 마세요")
""" """
# print("대기번호 : 1")
# print("대기번호 : 2")
# print("대기번호 : 3")
# print("대기번호 : 4")

# for waiting_no in [1,2,3,4]:
#     print("대기번호 : {0}".format(waiting_no))

# for waiting_no in range(1, 5): # 0,1,2,3,4
#     print("대기번호 : {0}".format(waiting_no))
starbucks = ["R", "Y", "U"]
for customer in starbucks:
    print("{0}, 커피가 준비되었습니다.".format(customer))
""" """
# customer = "R"
# index = 
# while index >= 1:
#     print("{0}, 커피가 준비되었습니다.{1}번 남았어요.".format(customer, index))
#     index-= 1
#     if index == 0:
#         print("커피는 폐기처분되었습니다.")

customer = "R"
person = "Unknown"

while person != customer :
    print("{0}, 커피가 준비되었습니다.".format(customer))
    person = input("이름이 어떻게 되세요?")
""" """
absent = [2, 5] # 결석
no_book = [7] # 책을 깜빡함
for student in range(1,11): # 1,2,3,4,5,6,7,8,9,10
    if student in absent:
        continue
    elif student in no_book:
        print("오늘 수업 여기까지. {0}는 교무실로 따라와".format(student))
        break
    print("{0}, 책을 읽어봐".format(student))
""" """
# 출석번호가 1 2 3 4 앞에 100을 붙이기로 함 101, 102, 103, 104
# students = [1,2,3,4,5]
# print(students)
# students = [i+100 for i in students]
# print(students)

# 이름을 길이로 변환
studnets = ["R", "RY", "RYU"]
students = [len(i) for i in studnets]
print(students)

# 학생 이름을 대문자로 변환
students = ["ryu", "ry", "Ru"]
students = [i.upper() for i in students]
print(students)
""" """
# Quiz) 당신은 cocoa 서비스를 이용하는 택시 기사님입니다.
# 50명의 승객과 매칭 기회가 있을 때, 총 탑승 승객 수를 구하는 프로그램을 작성하지오.

# 조건1 : 승객별 운행 소요 시간은 5 ~ 50분 사이의 난수로 정해집니다.
# 조건2 : ekdtlsdms thdy tlrks 5 ~ 15분 사이의 승객만 매칭해야 합니다.

# (출력문 예제)
# [0] 1번째 손님 (소요시간 : 15분)
# [ ] 2번째 손님 (소요시간 : 50분)
# [0] 3번째 손님 (소요시간 : 5분)
# ...
# [ ] 50번째 손님 (소요기간 : 16분)

# 총 탑승 승객 : 2분

from random import *
cnt = 0 # 총 탑승 승객
for i in range(1, 51): # 1 ~ 50 이라는 수 (승객)
    time = randrange(5,51) # 5분 ~ 50분 소요 시간
    if 5 <= time <= 15:
        print("[0] {0}번째 손님 (소요시간 : {1}분)".format(i, time))
        cnt += 1
    else:
        print("[ ] {0}번째 손님 (소요시간 : {1}분)".format(i, time))
print("총 탑승 승객 : {0} 분".format(cnt))
""" """
def open_account():
    print("새로운 계좌가 생성되었습니다.")
def deposit(balance, money):
    print("입금이 완료되었습니다. 잔액은 {0} 원 입니다.".format(balance + money))
    return balance + money
def withdraw(balance, money):
    if balance >= money:
        print("출금이 완료 되었습니다. 잔액은 {0}원 입니다.".format(balance - money))
        return balance - money
    else:
        print("출금이 완료되지 않았습니다. 잔액은 {0}원 입니다.".format(balance))
        return balance
def withdraw_night(balance, money):
    commission = 100
    return commission, balance -money - commission

balance = 0 # 잔액
balance = deposit(balance, 1000)
balance = withdraw(balance, 500)
commision, balance = withdraw_night(balance, 500)
print("수수료는 {0}원이며, 잔액은 {1}원 입니다.".format(commission, balance))
""" """
# def profile(name, age, main_lang):
#     print("이름 : {0}\t나이 : {1}\t주 사용언어 : {2}"\
#         .format(name, age, main_lang))
    
# profile("R", 20, "파이선")
# profile("y", 21, "자바")

def profile(name, age = 17, main_lang = "파이선"):
    print("이름 : {0}\t나이 : {1}\t주 사용언어 : {2}"\
        .format(name, age, main_lang))
profile("R")
profile("y")
""" """
def profile(name, age, main_lang):
    print("이름 : {0}\t나이 : {1}\t주 사용언어 : {2}"\
        .format(name, age, main_lang))
profile(main_lang = "자바", age = 25, name = "김태호")
""" """
# def profile(name, age, lang1, lang2, lang3, lang4, lang5):
#     print("이름 : {0}\t나이 : {1}\t".format(name, age), end = " ")
#     print(lang1, lang2, lang3, lang4, lang5)

# profile("유재석", 20, "P", "java", "c", "C++", "C#")
# prpfile("김태호", 25, "kotlin", "switf", "", "", ""
def profile(name, age, *lang):
    print("이름 : {0}\t나이 : {1}\t".format(name, age), end = " ")
    for lang in lang:
        print(lang, end = " ")
    print()
    
profile("유재석", 20, "P", "java", "c", "C++", "C#", "javascript")
profile("김태호", 25, "kotlin", "switf")
""" """
gun = 10

def checkpoint(soldiers):
    global gun
    gun = gun - soldiers
    print("[함수 내] 남은 총 : {0}".format(gun))
    
print("전체 총 : {0}".format(gun))
checkpoint(2) # 2명이 경계 근무 나감
print("남은 총 : {0}".format(gun))
""" """
def std_weight(height, gender):
    if gender == "male":
        return height**2*22
    else:
        return height**2*21

height = 190
gender = "남자"
weight = std_weight(height/100, gender)
print("키 {0}cm {1}의 표준 체중은 {2}kg입니다.".format(height, gender, weight))
""" """
print("Python", "java", "javascript", sep=" ", end="\n")
import sys
print("Python", "Java", file=sys.stdout)
print("Python", "Java", file=sys.stderr)
"""

import time

while True:
    # 타이머 시작점
    start = input("Enter를 누르면 타이머를 시작합니다.(Ctrl+c를 누르면 중지됩니다.)")
    begin = time.time()

    # 타이머 종료점
    stop = input("Enter를 누르면 측정을 종료합니다.")
    end = time.time()

    # 시간차
    result = end - begin

    # 여기서 round는 파이썬에서 소수점 자리수 조절에 활용됩니다.
    result = round(result, 3)
    print("시작 후", result, "초의 시간이 흘렀습니다.")
