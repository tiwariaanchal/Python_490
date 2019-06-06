file_name = input("Enter the file name")
infile = open(file_name, 'r')
my_dict = {}
line = infile.readline().lower()
word = line.split(" ")
for i in word:
    count = my_dict.get(i, 0)
    my_dict[i] = count + 1

my_list = my_dict.keys()

for x in my_list:
    print(x, my_dict[x])


