from collections import Counter
import sys

def print_ratios(d):
    c = Counter(d)
    total=sum(c.values())
    for k in sorted(c.keys()):
        print(k,c[k]/total)

data = [x.strip() for x in open(sys.argv[1]).readlines()]

if len(sys.argv)>3:
    part1_len, part2_len = map(int, sys.argv[2:4])
    print_ratios(data[:part1_len])
    print()
    print_ratios(data[-part2_len:])
else:
    print_ratios(data)
