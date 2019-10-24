from utils import load_file, write_file
from scipy import stats
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")

    args = parser.parse_args()

    if args.d == 'mnist':
        # prob_score = './sa/prob_mnist.txt'
        # prob_score = load_file(path_file=prob_score)
        # prob_score = [float(s) for s in prob_score]

        prob_score = './sa/dsa_mnist.txt'
        prob_score = load_file(path_file=prob_score)
        prob_score = [float(s) for s in prob_score]

        lsa_score = './sa/lsa_mnist.txt'
        lsa_score = load_file(path_file=lsa_score)
        lsa_score = [float(s) for s in lsa_score]

    if args.d == 'cifar':
        # prob_score = './sa/prob_cifar.txt'
        # prob_score = load_file(path_file=prob_score)
        # prob_score = [float(s) for s in prob_score]

        prob_score = './sa/dsa_cifar.txt'
        prob_score = load_file(path_file=prob_score)
        prob_score = [float(s) for s in prob_score]

        lsa_score = './sa/lsa_cifar.txt'
        lsa_score = load_file(path_file=lsa_score)
        lsa_score = [float(s) for s in lsa_score]

    print('==> Kolmogorov-Smirnov statistic test:')
    print(stats.ks_2samp(prob_score, lsa_score))
    print('==> T-test:')
    print(stats.ttest_ind(prob_score, lsa_score))
    print('==> Spearmanr-test:')
    print(stats.spearmanr(prob_score, lsa_score))

    