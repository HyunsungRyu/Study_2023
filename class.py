# print("Hello World")
# print("Hello")
# print("안녕하세요")


# print() # 줄바꿈
# print("안녕", "hello", "gg")
# print("안녕" + "hello" + "gg")


# x = 10
# print(x) # 10진수
# # 진법 변환 함수
# print(hex(x)) # 16진수 변환
# print(oct(x)) # 8진수 변환
# print(bin(x)) # 2진수 변환


# print("안녕하세요")
# print('hello')
# print("안녕하세요", 'hello')
# print("안녕하세요"+'hello')


# print()
# print(10)
# print(3.141592)
# print(-10)
# print(10, 20, -3.14)
# print(10+20+3.14)


# print()
# print('10+20')
# print(10+20)
# # print("10" + 20) # error
# print("10" + str(20))
# print(int("10")+20)


# print()
# print(True)
# print(False)
# print(5>10)
# print(5<= 10)


# print("hello" * 3)
# a = 3
# print("hello" * a)


# print("성별은 \"남자\"입니다.")
# print("성별은 '남자'입니다.")


# print("""
# 오늘은 금요일
# 집에 가는날
# 배고프다
# """)
# print("""\
# 오늘은 금요일
# 집에 가는날
# 배고프다\
# """)


# print('-'*30)
# print("""
# 안녕하세요.
# 파이썬 기초부터 시작하는 빅데이터 분석
# 모두 반가워요!!
# """)
# print('-'*30)
# # 줄바꿈 하고 싶지 않다면 \기호 사용하기
# print("""\
# 안녕하세요.
# 파이썬 기초부터 시작하는 빅데이터 분석
# 모두 반가워요!!\
# """)


# print("\t<<개인정보 출력>>")
# print("-"*30)
# print("\n\t이름 : 홍길동\n\t나이 : 17\n\t성별 : \"남\"\n\t키 : 170.5\n\t몸무게 : 77.6")


# name = "홍길동"
# age = 17
# gender = "남"
# height = 170.521
# weight = 77.6758
# #name = "홍길동";age = 17;gender = "남";height = 170.521;weight = 77.6758
# # name, age, gender, height, weight = "홍길동", 17, "남", 170.521, 77.6758


# print(name, age, gender, height, weight)
# print(type(name))
# print(type(age))
# print(type(gender))
# print("""\
# type(height)
# type(weight)\
# """)


# print(f"이름 : {name}")
# print(f"나이 : {age}")
# print(f"성별 : {gender}")
# print(f"키 : {height:.2f}")
# print(f"몸무게 : {weight:.1f}")


# x, y = 10, 3
# print(f"10 + 3 = {x + y}")
# print(f"10 - 3 = {x-y}")
# print(f"10 * 3 = {x * y}")
# print(f"10 / 3 = {x / y}")
# print(f"10 % 3 = {x % y}")
# print(f"10 // 3 = {x // 3}")
# print(f"10 ** 3 = {x ** y}")


# kor, math, eng = 100, 78, 66
# tot = kor + math + eng
# avg = tot / 3
# print(f"""\
# \t학점 계산기


# \t국어 성적 : {kor}
# \t수학 성적 : {math}
# \t영어 성적 : {eng}
# \t총점 : {tot}
# \t평균 : {avg:.2f}\
# """)


# x = input("정수 입력 : ") # input함수로 입력받은 변수는 str
# print("출력 결과 : " + x)


# m = int(input("교환할 돈 : "))
# print(f"\n500원짜리 : {m // 500}개");m %= 500
# print(f"\n500원짜리 : {m // 100}개");m %= 100
# print(f"\n500원짜리 : {m // 50}개");m %= 50
# print(f"\n500원짜리 : {m // 10}개");m %= 10
# print(f"\n바꾸지 못한 잔돈 : {m}원")


# year = int(input("연도 입력 : "))
# if ((year % 4) == 0 and (year % 100) != 0) or (year % 200) == 0:
#     print(f"{year}년은 윤년입니다.")
# else :
#     print(f"{year}년은 윤년이 아닙니다.")


# x = 0
# if x > 0:
#     print("양수")
#     print("월요일")
# elif x == 0:
#     print("zero")
# else:
#     print("음수")


# num = int(input("정수를 입력하세요. : "))
# if (num == 0):
#     print("0 입니다.")
# elif (num % 2 == 0):
#     print("짝수입니다.")
# else:
#     print("홀수입니다.")


# from random import *
# x = randint(1, 10)
# print(x)


# from random import *
# x = randint(1, 10)
# a = int(input("1부터 10까지 숫자를 입력하세요 : "))
# if (a == x):
#     print("정답입니다.")
# elif (a < x):
#     print(f"{a}보다 큽니다.")
# else:
#     print(f"{a}보다 작습니다.")
# print(f"answer : {x}")


# from random import *
# x = randint(0,4)
# y = randint(5, 9)
# z = randint(10, 14)
# ax = int(input("0~4 : "))
# ay = int(input("5~9 : "))
# az = int(input("10~14 : "))
# count = 0
# if (x == ax):
#     count += 1
# if (y == ay):
#     count+= 1
# if (z == az):
#     count+=1
# print(f"총 {count}번 맞췄습니다. 컴퓨터는 {x}, {y}, {z}를 냈습니다.")


# age = int(input("나이 입력 : "))
# height = int(input("키 입력 : "))
# if (height >= 120 and age >= 10):
#     print("탑승 가능")
# elif (height >= 120 and age >= 7):
#     print("보호자 동반 탑승 가능")
# else:
#     print("탑승 불가능")


# mm = 10000
# print("""\
# ---<<은행 프로그램>>---
# A. 예금 B. 출금 C. 잔고 확인
# """)
# key = input("원하시는 메뉴를 선택하세요 : ")
# if (key == "A" or key == "a"):
#     tm = int(input("얼마를 입급하시겠습니까? : "))
#     mm += tm
# elif (key == "B" or key == "b"):
#     tm = int(input("얼마를 출급하시겠습니까? : "))
#     if (tm > mm):
#         print("삑! 잔액부족입니다.")
#     else:
#         mm - tm
# elif (key == "C" or key == "c"):
#     print(f"현재 금액 : {mm}")
# else:
#     print("잘못된 접근입니다.")

# import math
# inHour = int(input("들어온 시간을 입력하시오 : "))
# inMinute = int(input("들어온 분을 입력하세요 : "))
# outHour = int(input("나간 시간을 입력하세요 : "))
# outMinute = int(input("나간 분을 입력하세요 : "))
# Minute =  ((outHour - inHour) * 60) + (outMinute - inMinute)
# fee = int(0)
# print(f"주차한 시간은 {Minute}분 입니다.")
# if (Minute <= 30):
#     fee = 2000
# elif (Minute > 120):
#     fee = 6500 + 400 * math.ceil((Minute - 120) / 10)
# elif (Minute > 30):
#     fee = 2000 + 500 * math.ceil((Minute - 30) / 10)
# print(f"최종 요금은 {fee}원 입니다.")


# a = 0
# b = int(input("정수 입력 : "))
# sum = 0
# even = 0
# while a <= b:
#     sum += a
#     if a % 2 == 0:
#         even += a
#     a += 1
# print(f"전체 합 : {sum}")
# print(f"짝수 합 : {even}")


# a = int(input("몇 단을 출력할까요? : "))
# b = 0
# while (b < 9):
#     b += 1
#     print(f"{a} * {b} = {a*b}")

# c = int(input("몇 단을 출력할까요? : "))
# d = 10
# while (d > 1):
#     d -= 1
#     print(f"{c} * {d} = {c*d}")


# i = 0
# while True:
#     print("*")
#     i += 1
#     if(i > 4):
#         break


# from random import *
# an = randint(1, 10)
# count = 0
# while True:
#     key = int(input("1부터 10 이하의 숫자를 맟줘보세요 : "))
#     if key == an:
#         count += 1
#         print(f"정답입니다. {count}번 만에 맞췄습니다.")
#         break
#     else:
#         print("다시")
#         count += 1


# print("""
# ----------------------------------------------
# <<비밀번호 맞추기>>
# ----------------------------------------------
#       """)
# APW = "asdf@123"
# password = 0
# count = 0
# while True:
#     password = input(">> 비밀번호를 입력하세요 : ")
#     if (password != APW):
#         print("잘못된 비밀번호입니다. 다시 입력해주세요.")
#         count+=1
#     else:
#         count+=1
#         print(f"맞춘 횟수는 : {count}입니다.")
#         break


# for i in range(5): # 0 1 2 3 4
#     print(f"hello {i}", end = ' ')
# print(f"\n{i}")
# for i in range(1,6): #  1 2 3 4 5
#     print(f"hello {i}", end = ' ')
# for i in range(1,10, 2): #  1 3 7 9
#     print(f"hello {i}", end = ' ')
# for i in range(10,1, -2): # 10 8 6 4 2
#     print(f"hello {i}", end = ' ')


# sum = 0
# for i in range (1, 101):
#     sum += i
# print(f"1부터 {i}까지의 합 : {sum}")


# sum = 0
# limit = int(input("정수 입력 : "))
# for i in range (1, limit+1):
#     sum += i
# print(f"1부터 {i}까지의 합 : {sum}")


# even = 0
# odd = 0
# for i in range(1, 101):
#     if i % 2 == 0:
#         even += i
#     else:
#         odd += i
# print(f"홀수합 : {odd}\n짝수합 : {even}")


# a = int(input("구구단 몇 단을 출력할까요? : "))
# for i in range(1,10):
#     print(f"{a} x {i} = {a * i}")
# print("\n")
# for i in range(9, 0, -1):
#     print(f"{a} x {i} = {a * i}")


# sum = 0
# for i in range(1, 11):
#     if i % 2 == 0:
#         sum += i
#     else:
#         sum -= i
#     print(i)
# print(f"sum = : {sum}")


# a = int(input("정수 입력 : "))
# if a > 2:
#     for i in range(2, a):
#         if a % i == 0:
#             print(f"{a}는 소수가 아닙니다.")
#             break
#         else:
#             continue
#     if a - 1 == i:
#         print(f"{a}는 소수입니다.")
# else:
#     print(f"{a}는 소수입니다.")


# for i in range(2, 10):
#     for j in range(2, 10):
#         print(f"{i} * {j} = {i * j}")


# sum = 0
# for i in range(1, 51):
#     sum += i
#     if (sum >= 100):
#         print(f"1부터 {i}까지 더할 때 {sum}으로 100을 넘게 된다.")
#         break
#     else:
#         continue


# sum = 0
# for i in range(1, 101):
#     if (i % 5 == 0):
#         continue
#     else:
#         sum += i
# print(f"5의 배수를 제외한 1부터 100까지의 수의 합은 {sum}입니다.")


# sum = 0
# for i in range(3333, 10000):
#     if i % 1234 == 0:
#         continue
#     else:
#         sum += i
#     if sum >= 100000:
#         sum -= i
#         break
# print(f"{sum}")


# for i in range(3, 101):
#     for j in range(2, i):
#         if(i % j == 0):
#             break
#         if( j == i-1):
#             print(i)


# for i in range(1, 6):
#     print("*"*i)


# for i in range(1, 6):
#     print(" "*(5-i)+"*"*i)


# for i in range(1, 6):
#     print(" "*(5-i)+"*"*(1+(i-1)*2))


# for i in range(1, 6):
#     print(" "*(i-1),end = "")
#     print("*"*(1+2*(5-i)))


# for i in range(1, 6):
#     print(" "*(5-i)+"*"*(1+(2*(i-1))))
# for i in range(1, 5):
#     print(" "*i+"*"*(1+2*(4-i)))


# for i in range(1, 6):
#     print(" "*(i-1),end = "")
#     print("*"*(1+2*(5-i)))
# for i in range(1, 5):
#     print(" "*(4-i)+"*"*(1+i*2))


# 리스트


# aa = [10, 20, 30]
# bb = [40, 50, 60]
# print(aa+bb)
# print(aa*3)


# aaa = [10, 20, 30, 40, 50, 60, 70]
# # 앞에서부터 2칸씩 건너뜀
# print(aaa[::2])
# # 뒤에서부터 2칸씩 건너뜀
# print(aaa[::-2])
# #뒤에서부터 1칸씩 건너뜀(역순으로 출력)
# print(aaa[::-1])
# # 첫번째부터 2칸씩 건너뜀
# print(aaa[1::2])


# # 삽입
# li = [10, 20, 30]
# li.append(40)
# li.append(50)
# print(li)


# li = [10, 20, 30]
# li.insert(1, 15)


# #제거
# li = [10, 20, 30, 40, 50]
# print(li)
# li.pop()
# li.pop(1)
# print(li)


# li = [10, 20, 30, 20, 50]
# print(li)
# li.remove(20)
# print(li)


# del li[0]
# print(li)


# li = [10, 2, 30, 4, 50]
# li2 = sorted(li)
# li.extend(li2)
# print(li2)
# li.reverse()
# print(li)


# li = [10, 20, 30, "홍길동"]
# print(li.index(20)) # 1
# print(li.index("홍길동")) # 3


# li = [10, 20, 30, 20]
# print(li.count(20))


# li = [10, 20, 30, 20]
# print(len(li)) # 4


# li = [10, 20, 30, 20]
# li.clear()
# print(li)


# li = [10, 20, 30, 20]
# li2 = li.copy()
# print(li2)


# li = [10, 20, 30, 20]
# print(20 in li) # True
# print(15 in li) # False

# if 15 in li:
#     print("15가 있다")
# else:
#     print("15가 없다.")
# if 15 not in li:
#     print("15가 있다")
# else:
#     print("15가 없다.")


# bakery = ["스콘", "케이크", "샌드위치"]
# print(f"메뉴 개수는 : {len(bakery)}개이다.")
# bakery.insert(0, "약과")
# bakery.append("햄치즈 샌드위치")
# bakery.append("에그샐러드 샌드위치")
# bakery.append("닭가슴살 샌드위치")
# bakery[3] = "불고기 샌드위치"
# bakery.remove("스콘")
# print(bakery)


# student = int(input("학생 수 입력 : "))
# score_list = []
# for i in range(student):
#     score = int(input(f"{i+1}번 점수를 입력하세요."))
#     score_list.append(score)
# print(score_list)
# print(f"{student}의 점수 : ", score_list)
# print(f"{student}의 최대 점수 : ", max(score_list))
# print(f"{student}의 죄소 점수 : ", min(score_list))
# print(f"{student}의 점수 합계 : ", sum(score_list))
# print(f"{student}의 점수 평균 : {sum(score_list)/student:.2f}")


# li = [10, 20, 30, 40, 50]
# for i in li:
#     print(i)

# # 1
# a = ["안녕하세요", "Hello", "HI", "bye"]
# for i in a:
#     print(i[0], end = " ")


# #  2
# a1 = [1, 2, 3, 4]
# a2 = []
# for i in range(len(a1)) :
#     a2.append(a1[i])
# print(a1)
# print(a2)


# # 3
# import random
# a = []
# for i in range(1, 11):
#     a.append(random.randint(1, 11))
# print(a[0])
# print(sum(a))


# print("%04d" % 876)
# print("%5s" % "CookBook")
# print("%1.1f" % 123.45)


# # 4
# a = ["alpha", "brove", "delta", "echo", "foxtrot", "golf", "hotel", "india"]
# b = []
# for i in a:
#     if len(i) == 5:
#         b.append(i)

# print(b)


# # 5
# a = [[1,2,3],[4,5],[6,7,8,9]]
# for i in a:
#     for  j in i:
#         print(j, end = " ")
#     print()


# # 6
# li = []
# li2 = []
# value = 1
# for i in range(3):
#     for j in range(4):
#         li.append(value)
#         value+=1
#     li2.append(li)
#     li = []
# print(li2)


# for i in range(3):
#     for j in range(4):
#         print(li[i][j], end = " ")
#     print()


# from random import *
# CA = []
# UA = []
# cnt = 0
# while(len(CA) < 6):
#     a = randint(1, 45)
#     if a not in CA:
#         CA.append(a)
# CA.sort()
# print(CA)
# while(len(UA) < 6):
#     a = int(input("1~45까지의 수를 입력하세요 : "))
#     if a <=45:
#         if a not in UA:
#             UA.append(a)
#         else: print("이미 입력된 번호입니다.")
#     else: print("1부터 45까지의 수를 입력하세요.")
# UA.sort()
# print(f"사용자 로또 입력 : {UA}")
# for i in range(6):
#     if UA[i] in CA:
#         cnt += 1
# print(f"{cnt}개 일치")
# if cnt >=3:
#     print(f"{7-cnt}등 당첨")
# else:
#     print("다음 기회에")


# from random import *
# cnt, s, b = 0, 0, 0
# CN = []
# while(len(CN) < 3):
#     a = randint(1,9)
#     if a not in CN:
#         CN.append(a)
# while(s < 3):
#     UN = []
#     while(len(UN) < 3):
#         a = int(input("숫자 3개 입력 : "))
#         if a not in UN:
#             UN.append(a)
#     s , b = 0, 0
#     for i in range(3):
#         if (UN[i] == CN[i]):
#             s+=1
#         elif (UN[i] in CN):
#             b+=1
#     print(f"{s} Strike, {b} Ball")
#     cnt += 1
# print(f"축하합니다!! {cnt}번만에 맞췄습니다.")


# a1 = (10, 20, 30)
# a2 = 10, 20, 30  # 소괄호 생략이 가능하다
# a3 = (10,)
# print(a1)
# print(a2)
# print(a3)

# a1 = 10, 20, 30
# print(a1[0])  # 10
# print(a1[-1])  # 30
# print(a1[0:2])  # 10, 20
# print(a1[1:])  # 20, 30

# for i in a1:
#     print(i, end=" ")
# for i in range(3):
#     print(a1[i], end=" ")

# a1 = (10, 20, 30)
# a1[1] = 15 # error
# a1.append(40) # error
# a2 = ("A", "B")
# print(a1 + a2)
# print(a1)
# print(a2)
# print(a2 * 2)
# del a2  # a2튜플 삭제
# li = list(a1)
# print(li)
# a2 = tuple(li)
# print(a2)


# dict_eng = {"apple": "사과", "peach": "복숭아", "banana": "바나나"}
# print(dict_eng["apple"])
# print(dict_eng["peach"])
# print(dict_eng["watermelon"]) # error

# dict_eng = {"apple": "사과", "peach": "복숭아", "banana": "바나나"}
# dict_eng["watermelon"] = "수박"
# print(dict_eng)
# dict_eng["melon"] = "메론"
# dict_eng["melon"] = "멜론"
# print(dict_eng)
# del dict_eng["watermelon"]
# print(dict_eng)

# if "strawberry" in dict_eng:
#     print(dict_eng["strawberry"])
# else:
#     print("존재하지 않는 키워드입니다.")

# x = 'strawberry'
# if x in dict_eng:
#     print(dict_eng[x])
# else:
#     print("존재하지 않는 키워드입니다.")

# get('key') : 딕셔너리에서 없는 키 값을 찾아야 할 경우
# dict_eng = {}
# x = dict_eng.get("strawberry")
# print(x)
# if x == None:
#     print("존재하지 않는 키워드")
# else:
#     print(x)


# print(dict_eng.keys())
# print(dict_eng.values())
# print(dict_eng.items())


# a1 = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
# # print(f"{a1[0]}\n{a1[1]}\n{a1[2]}")
# for i in a1:
#     for j in i:
#         print(j, end=" ")
#     print()

# singer = {}
# singer["이름"] = "트와이스"
# singer["구성원 수 "] = 9
# singer["데뷔"] = "서바이벌 식스틴"
# singer["대표곡"] = "cheer up"
# singer_list_keys = list(singer.keys())
# print(singer_list_keys)
# singer_list_values = list(singer.values())
# print(singer_list_values)
# # for i in range(4):
# #     print(f"{singer_list_keys[i]}-->{singer_list_values[i]}")
# for i in singer.keys():
#     print(f"{i}-->{singer[i]}")


# pets = [{"name": "구름", "age": 5}, {"name": "초코", "age": 7}, {"name": "뭉치", "age": 13}]
# print("우리 동네 애완 동물들")
# for i in pets:
#     print(f"{i['name']} {i['age']}살")


# numbers = [1, 2, 3, 4, 5, 1, 3, 5, 1]
# counter = {}
# for i in numbers:
#     if i not in counter:
#         counter[i] = 1
#     else:
#         counter[i] += 1
# print(counter)


# key_list = ["name", "hp", "mp", "level"]
# value_list = ["기사", 200, 30, 5]
# character = {}
# cnt = 0
# for i in key_list:
#     character[i] = value_list[cnt]
#     cnt += 1
# print(character)


# en_dict = {"apple": "사과", "peach": "복숭아", "banana": "바나나", "lemon": "레몬", "melon": "멜론"}
# print(f"현재 단어장 : {en_dict}")
# while True:
#     key = 0
#     user_an = input("검색할 영어 단어를 입력하세요 : ")
#     if user_an not in en_dict:
#         print("검색한 단어는 등록되어 있지 않습니다\n\n")
#         key = input("단어를 새로 등록하시겠습니까? y/n으로만 답하세요 : ").lower()
#         if key == "y":
#             new_en = input("단어 뜻을 입력하세요 : ")
#             en_dict[user_an] = new_en
#             key = input("계속하시겠습니까? y/n으로만 답하세요 : ").lower()
#             if key == "y":
#                 continue
#             else:
#                 break
#         else:
#             continue
#     else:
#         print(f"{user_an}의 뜻은 '{en_dict[user_an]}'입니다.")
#         key = input("계속 하시겠습니까? y/n으로만 답하세요 : ").lower()
#         if key == "y":
#             continue
#         else:
#             break
# print(f"현재 단어장 : {en_dict}")
# # dict_eng =  {"apple": "사과", "peach": "복숭아", "banana": "바나나", "lemon": "레몬", "melon": "멜론"}
# # print("현재 단어장", dict_eng)
# # while True:
# #     print()
# #     user = input(">> 검색할 영어 단어를 입력하세요 : ").lower()
# #     if user in dict_eng:
# #         print(f">>{user}의 뜻은 '{dict_eng[user]}'")
# #     else:
# #         print("검색한 단어는 등록되어 있지 않습니다\n")
# #         register = input("단어를 새로 등록하시겠습니까? y / n 으로만 답하세요 : ").lower()
# #         if register == 'y':
# #             mean = input("단어 뜻을 입력하세요 : ")
# #             dict_eng[user] = mean
# #     choice = input("계속 하시겠습니까? y / n 으로만 답하세요 : ").lower()
# #     if choice == 'n':
# #         break
# # print("\n>>추가된 단어장 : ", dict_eng)


# import operator

# trainDic = {"Tomas": "토마스", "Edward": "에드워드", "Henry": "헨리"}
# trainList = []
# trainList = sorted(trainDic.items(), key=operator.itemgetter(0)) # 영어 "Tomas"
# trainList = sorted(trainDic.items(), key=operator.itemgetter(1)) # 한글 "토마스"
# print(trainList)


# # set
# s1 = {1, 2, 3, 4, 5, 1, 2, 3}
# print(s1) # 중복 x


# salesList = ["삼각김밥", "도시락", "바나나", "도시락", "도시락", "삼각김밥"]
# print(set(salesList))


# s1 = {1, 2, 3, 4, 5, 1, 2}
# s2 = {2, 4, 6, 7}

# print(s1 & s2)  # 교집합 {2, 4}
# print(s1.intersection(s2))  # 교집합 {2, 4}

# print(s1 | s2)  # 합집합 {1, 2, 3 ,4, 5, 6, 7}
# print(s1.union(s2))  # 합집합 {1, 2, 3 ,4, 5, 6, 7}

# print(s1 - s2)  # 차집합 {1, 3, 5}
# print(s1.difference(s2))  # 차집합 {1, 3, 5}

# print(s1 ^ s2)  # 대칭차집합 {1, 3, 5, 6, 7}
# print(s1.symmetric_difference(s2))  # 대칭차집합 {1, 3, 5, 6, 7}


# # Comprehension
# li = [i for i in range(1, 6)]
# print(li)
# li = [i * i for i in range(1, 6)]
# print(li)
# li = [i for i in range(1, 21) if i % 3 == 0]
# # print(li)
# myData = [[n * m for n in range(1, 3)] for m in range(2, 4)][[2, 4], [3, 6]]
# print(myData)


# foods = ["떡볶이", "마라탕", "라면", "피자", "치킨"]
# sides = ["오뎅", "꿔바로우", "김치"]
# print(zip(foods, sides))
# print(list(zip(foods, sides)))
# print(dict(zip(foods, sides)))

# for i, j in zip(foods, sides):
#     print(i, "-->", j)


# num = input("주민번호 입력(중간에 '-'기호 사용) : ")
# gen = num[7]
# if gen == "3" or gen == "1":
#     print("남자입니다.")
# elif gen == "2" or gen == "4":
#     print("여자입니다.")
# else:
#     print("error")


# sen = input("문자열 입력 : ")
# for i in range(len(sen)):
#     if i % 2 == 0:
#         print(sen[i], end="")
#     else:
#         print("#", end="")


# re_str = ""
# str = input("문자열을 입력하세요 : ")
# for i in range(len(str)):
#     re_str = str[::-1]  # re_str += str[len(str) - (i + 1)]
# print(re_str)
# # 동연이 코드
# # re_str = ""
# # str = input("문자열을 입력하세요 : ")
# # for i in str[::-1]:
# #     re_str += i
# # print(re_str)


# s1 = "python is EZ"
# print(s1.upper())
# print(s1.lower())
# print(s1.swapcase())
# print(s1.title())
# s1 = "파이썬 공부 화이팅! 4반 화이팅!"
# print(s1.count("화이팅"))
# print(s1.find("공부"), s1.find("화이팅"), s1.find("자바"))
# print(s1.rfind("공부"), s1.rfind("화이팅"), s1.rfind("자바"))  # 오른쪽부터
# print(s1.index("공부"), s1.index("화이팅"))  # s1.index("자바")) # 없는 것을 찾을때 오류
# print(s1.rindex("공부"), s1.rindex("화이팅"))
# print(s1.startswith("파이썬"), s1.endswith("자바"))
# print(s1.startswith("자바"), s1.endswith("화이팅!"))


str = "aacdAABbE"
str = str.lower()
print(str)
most = 0
for i in range(0 + 97, 26 + 97):
    cnt = str.count(chr(i))
    if most == cnt:
        pre = "?"
    elif most < cnt:
        pre = chr(i)
        most = cnt
print(pre)
# 동연이 코드
# str = "abcdAAdbBE"
# li = []
# cnt = 0
# str = str.lower()
# for i in range(26):
#     li.append(0)
# for j in range(26):
#     cnt = str.count(chr(j + 97))
#     li[j] = cnt
#     cnt = 0
# if li.count(max(li)) > 1:
#     print("?")
# else:
#     i = li.index(max(li))
#     print(chr(i + 97))
