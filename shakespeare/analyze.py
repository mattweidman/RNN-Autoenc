# find list of characters used
filename = "shakes_out.txt"
with open(filename, 'r') as f:
    text = f.read()
chars = sorted(set(text))
print(chars)

# find longest line
with open(filename, 'r') as f:
    longest_line = ""
    for line in f.readlines():
        if len(line) > len(longest_line):
            longest_line = line
print("longest line:")
print(longest_line)
