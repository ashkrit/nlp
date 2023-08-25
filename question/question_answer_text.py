from transformers import pipeline 
import sys 

print("Speak your mind > ")
question_answerer = pipeline("question-answering")

data = """
Visa Inc. (/ˈviːzə, ˈviːsə/; stylized as VISA) is an American multinational financial services corporation headquartered in San Francisco, California.[1][4] It facilitates electronic funds transfers throughout the world, most commonly through Visa-branded credit cards, debit cards and prepaid cards.[5] Visa is one of the world's most valuable companies.
Visa does not issue cards, extend credit or set rates and fees for consumers; rather, Visa provides financial institutions with Visa-branded payment products that they then use to offer credit, debit, prepaid and cash access programs to their customers. In 2015, the Nilson Report, a publication that tracks the credit card industry, found that Visa's global network (known as VisaNet) processed 100 billion transactions during 2014 with a total volume of US$6.8 trillion.[6]
Visa was founded in 1958 by Bank of America (BofA) as the BankAmericard credit card program.[7] In response to competitor Master Charge (now Mastercard), BofA began to license the BankAmericard program to other financial institutions in 1966.[8] By 1970, BofA gave up direct control of the BankAmericard program, forming a cooperative with the other various BankAmericard issuer banks to take over its management. It was then renamed Visa in 1976.[9]
Nearly all Visa transactions worldwide are processed through the company's directly operated VisaNet at one of four secure data centers, located in Ashburn, Virginia; Highlands Ranch, Colorado; London, England; and Singapore.[10] These facilities are heavily secured against natural disasters, crime, and terrorism; can operate independently of each other and from external utilities if necessary; and can handle up to 30,000 simultaneous transactions and up to 100 billion computations every second.[6][11][12]
Visa is the world's second-largest card payment organization (debit and credit cards combined), after being surpassed by China UnionPay in 2015, based on annual value of card payments transacted and number of issued cards.[13] However, because UnionPay's size is based primarily on the size of its domestic market in China, Visa is still considered the dominant bankcard company in the rest of the world, where it commands a 50% market share of total card payments.[13]
"""

print(data)
print("Speak your mind > ")
for line in sys.stdin:
    result = question_answerer(question=line.strip(),context=data)
    print(result)
    print("Speak your mind > ")