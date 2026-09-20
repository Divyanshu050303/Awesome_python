from collections import defaultdict

expenses = [
    ("Food", 250),
    ("Travel", 500),
    ("Food", 150),
    ("Shopping", 1200),
]

amount =[amt for _, amt in expenses]

total= sum(amount)
highest =max(expenses, key=lambda x:x[1])
lowest =min(expenses, key=lambda x:x[1])
average=total/len(expenses)

by_category=defaultdict(int)
print(by_category)
for name, amt in expenses:
    by_category[name]+=amt

print("Total:", total)
print("Highest:", highest)
print("Lowest:", lowest)
print("By category:", dict(by_category))
print("Average:", average)

