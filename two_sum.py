nums = [2,7,11,15]
right=len(nums)
seen={}
target=9
for i, num in enumerate(nums):
    complement = target - num
    print(f"num: {num}, complement: {complement}, seen: {seen}")

    if complement in seen:
        print([seen[complement], i])

    seen[num] = i
