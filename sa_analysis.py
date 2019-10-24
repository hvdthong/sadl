import argparse
from utils import load_file
from statistics import median

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")

    parser.add_argument("--lsa", "-lsa", help="Likelihood Surprise Adequacy Score", action="store_true")
    parser.add_argument("--dsa", "-dsa", help="Distance-based Surprise Adequacy Score", action="store_true")

    args = parser.parse_args()
    print(args)

    if args.d == "mnist" or args.d == 'cifar':
        if args.lsa:
            lsa = load_file('./sa/lsa_%s.txt' % args.d)
            lsa = [float(s) for s in lsa]
            print('./sa/lsa_%s.txt' % args.d)
            print('Average: %.3f -- Max: %.3f -- Min: %.3f -- Median: %.3f' % (sum(lsa)/len(lsa), max(lsa), min(lsa), median(lsa)))
        elif args.dsa:
            dsa = load_file('./sa/dsa_%s.txt' % args.d)
            dsa = [float(s) for s in dsa]
            print('./sa/dsa_%s.txt' % args.d)
            print('Average: %.3f -- Max: %.3f -- Min: %.3f -- Median: %.3f' % (sum(dsa)/len(dsa), max(dsa), min(dsa), median(dsa)))
            
