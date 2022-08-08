import argparse

parser = argparse.ArgumentParser(description='hello jaejung', add_help=True, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--name', type=str, default='jaejung', help='name')
parser.add_argument('--integer', type=int, help='intr')
print(parser)

args = parser.parse_args()
print(type(args))
print(args)
print(args.name)
print(args.integer)
print()

if __name__ == '__main__':
    print('main')
# else:
#     print("not main")

print('='*30)
