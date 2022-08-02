total = 634112
V1 = 0.9547613537050023
V2 = 0.4851290519156297
V3 = 0.44758605680293084

oran = (V2 / (1-V2)) + 1 + (V1 / (1-V1)) + (((1-V3) * V2) / (V3 * (1-V2)))
b = total / oran
a = b * V2 / (1-V2)
c = a * (1-V3) / V3
d = total - (a + b + c)
print(a)
print(b)
print(c)
print(d)
print( (a+b) / total)
print( (a+c) / total)
print((a+b)/ (a+c))