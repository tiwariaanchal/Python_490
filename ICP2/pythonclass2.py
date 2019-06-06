n = int(input("How many students are there?"))
weights_in_lbs = []
weights_in_kgs = []
for i in range(n):
    x = float(input("Enter the weight"))
    weights_in_lbs.append(x)
    x = x * 0.453
    weights_in_kgs.append(x)
print(weights_in_lbs)
print(weights_in_kgs)