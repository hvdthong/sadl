from utils import load_file 

if __name__ == '__main__':
    correct = './figures_QA/correct_info_cifar'
    correct = load_file(correct)

    lsa_score = [float(l.split('\t')[2]) for l in correct]
    prob_score = [float(l.split('\t')[1]) for l in correct]
    condition = ['null' for l, p in zip(lsa_score, prob_score) if l >= 2000 and p <= 0.75]
    print('correct instance: ', len(condition))

    incorrect = './figures_QA/incorrect_info_cifar'
    incorrect = load_file(incorrect)

    lsa_score = [float(l.split('\t')[2]) for l in incorrect]
    prob_score = [float(l.split('\t')[1]) for l in incorrect]
    condition = ['null' for l, p in zip(lsa_score, prob_score) if l >= 2000 and p <= 0.75]
    print('incorrect instances: ', len(condition))