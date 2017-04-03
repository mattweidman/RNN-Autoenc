# find list of characters used
filename = "shakes_out.txt"
with open(filename, 'r') as f:
    text = f.read()
chars = sorted(set(text))
print(chars)
