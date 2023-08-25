from transformers import pipeline
import sys
import pandas as pd

pipe = pipeline("table-question-answering",
                model="google/tapas-medium-finetuned-wtq")

data = {
    "customer": ["C001", "COO2", "COO2", "C002"],
    "merchant": ["MCD", "SubWay", "Burger King", "MCD"],
    "amount": ["3.99", "6.78", "2.99", "9.99"],
}

query = [
    {"table": data, "query": "how much amount c001 is spending"},
    {"table": data, "query": "who is top spender ?"}
]

content = {"header": ["Year", "Division", "League", "Regular Season", "Playoffs", "Open Cup", "Avg. Attendance"],
           "rows": [["2001", "2", "USL A-League", "4th, Western", "Quarterfinals", "Did not qualify", "7,169"],
                    ["2002", "2", "USL A-League", "2nd, Pacific",
                        "1st Round", "Did not qualify", "6,260"],
                    ["2003", "2", "USL A-League", "3rd, Pacific",
                        "Did not qualify", "Did not qualify", "5,871"],
                    ["2004", "2", "USL A-League", "1st, Western",
                        "Quarterfinals", "4th Round", "5,628"],
                    ["2005", "2", "USL First Division", "5th",
                        "Quarterfinals", "4th Round", "6,028"],
                    ["2006", "2", "USL First Division", "11th",
                        "Did not qualify", "3rd Round", "5,575"],
                    ["2007", "2", "USL First Division", "2nd",
                        "Semifinals", "2nd Round", "6,851"],
                    ["2008", "2", "USL First Division", "11th",
                        "Did not qualify", "1st Round", "8,567"],
                    ["2009", "2", "USL First Division", "1st",
                        "Semifinals", "3rd Round", "9,734"],
                    ["2010", "2", "USSF D-2 Pro League", "3rd, USL (3rd)", "Quarterfinals", "3rd Round", "10,727"]]
           }


# print(content["header"])

# print(content["rows"])

column_store = {

}

for headerIndex in range(len(content["header"])):
    col_name = content["header"][headerIndex]
    # print("Loading index " , headerIndex , col_name)
    # print(content["header"][headerIndex])
    col_vals = []
    for val in content["rows"]:
        col_vals.append(val[headerIndex])

    # print(col_vals)
    column_store[col_name] = col_vals

# print(column_store)


###

print("Speak your mind > ")
for input in sys.stdin:
    query = [{"table": column_store, "query": input.strip()}]
    result = pipe(query)
    print(result)
    for r in result:
        print(r)
    print("Speak your mind > ")
