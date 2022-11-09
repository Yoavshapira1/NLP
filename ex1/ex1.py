def increase_counter(dict, val):
    try:
        dict[val] += 1
    except:
        dict[val] = 1

if __name__ == "__main__":
    d = {}
    try:
        d[0] += 1
    except:
        d[0] = 0
    print(d[0])