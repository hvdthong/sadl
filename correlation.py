import argparse
from utils import load_file, write_file
from scipy import stats
from threshold_sa import normalize_sa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--write", "-write", help="Write the Probability Score", action="store_true"
    )
    parser.add_argument(
        "--adv", "-adv", help="Working on the Adversarial Examples", action="store_true"
    )

    args = parser.parse_args()

    if args.d == 'mnist':
        if args.adv:
            path_dsa_score = './sa/dsa_adversarial_mnist.txt'
            dsa_score = normalize_sa(load_file(path_file=path_dsa_score))

            path_prob_score = './sa/prob_adv_mnist.txt'
            prob_score = load_file(path_file=path_prob_score)
            prob_score = [float(s) for s in prob_score]       
        else:
            path_dsa_score = './sa/dsa_mnist.txt'
            dsa_score = normalize_sa(load_file(path_file=path_dsa_score))

            path_prob_score = './sa/prob_mnist.txt'
            prob_score = load_file(path_file=path_prob_score)
            prob_score = [float(s) for s in prob_score]       

    if args.d == 'cifar':
        if args.adv:
            path_dsa_score = './sa/dsa_adversarial_cifar.txt'
            dsa_score = normalize_sa(load_file(path_file=path_dsa_score))

            path_prob_score = './sa/prob_adv_cifar.txt'
            prob_score = load_file(path_file=path_prob_score)
            prob_score = [float(s) for s in prob_score]       
        else:
            path_dsa_score = './sa/dsa_cifar.txt'
            dsa_score = load_file(path_file=path_dsa_score)
            dsa_score = normalize_sa(load_file(path_file=path_dsa_score))

            path_prob_score = './sa/prob_cifar.txt'
            prob_score = load_file(path_file=path_prob_score)
            prob_score = [float(s) for s in prob_score]

    if args.write == True:
        if args.adv:
            path_write_file = './sa/dsa_adv_normalize_%s.txt' % (args.d)
            write_file(path_file=path_write_file, data=dsa_score)
        else:
            path_write_file = './sa/dsa_normalize_%s.txt' % (args.d)
            write_file(path_file=path_write_file, data=dsa_score)

    print('==> Kolmogorov-Smirnov statistic test:')
    print(stats.ks_2samp(dsa_score, prob_score))
    print('==> T-test:')
    print(stats.ttest_ind(dsa_score,prob_score))
    print('==> Spearmanr-test:')
    print(stats.spearmanr(dsa_score,prob_score))

