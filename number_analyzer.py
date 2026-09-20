nums=[int(x) for x in input("Enter numbers: ").split()]

def is_prime(n):
    if n<2:
        return False
    for d in range(2, int(n**0.5)+1):
        if n%d==0:
            return False
    return True


evenNumbers=[]
oddNumbers=[]
primeNumbers=[]
duplicates=[]


for i in nums:
    if(i%2==0):
        evenNumbers.append(i)
    else:
        oddNumbers.append(i)

    if is_prime(i):
           primeNumbers.append(i) 
    if nums.count(i)>1 and i not in duplicates:
        duplicates.append(i)

print("even number", evenNumbers)                
print("odd number", oddNumbers)                
print("duplicate number", duplicates)                
print("Prime number", primeNumbers)                
print("maximum number", max(nums))                
print("minmum number", min(nums))                
print("average number", sum(nums)/len(nums))                
