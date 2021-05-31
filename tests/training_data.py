import re


def plot_loss(str_log):

    train_regex = re.compile('TRAIN LOSS: [0-9.]{5}')
    train_loss = train_regex.findall(str_log)
    train_loss = [float(x[-5:]) for x in train_loss]

    val_regex = re.compile('VALID LOSS: [0-9.]{5}')
    val_loss = val_regex.findall(str_log)
    val_loss = [float(x[-5:]) for x in val_loss]

    global_steps_train = list(range(0, len(train_loss)*312, 312))
    global_steps_val = list(range(0, len(val_loss)*312, 312))


def plot_kacc(str_log):
    regex = re.compile('@1: [0-9.]{5}')
    acc = regex.findall(str_log)
    acc = [float(x[-5:]) for x in acc]

    global_steps = list(range(0, len(acc)*312, 312))

    plt.figure(figsize=(8, 4))
    plt.ylim([0.0, 0.5])
    plt.plot(global_steps, acc, "-", label="acc@1")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("acc@1")
    plt.savefig("figures/kacc_graph.png")


def barplot_kacc(str_log):
    regex = re.compile('@[0-9]{1,2}: [0-9.]{5}')
    acc = regex.findall(str_log)
    acc = [float(x[-5:]) for x in acc]

    import seaborn as sns
    import pandas as pd
    sns.set_theme(style="whitegrid")
    acc_df = pd.DataFrame(
        {'acc@1': [acc[-4]],
         'acc@3': [acc[-3]],
         'acc@5': [acc[-2]],
         'acc@10': [acc[-1]]}
        )
    ax = sns.barplot(data=acc_df, ci=None)
    ax.figure.savefig("figures/barplot.png")


def barplot_kacc_random(str_log):
    regex = re.compile('@[0-9]{1,2}: [0-9.]{5}')
    acc = regex.findall(str_log)
    acc = [float(x[-5:]) for x in acc]

    import seaborn as sns
    import pandas as pd
    sns.set_theme(style="whitegrid")
    acc_df = pd.DataFrame(
        {'acc@1': [acc[-4]],
         'acc@3': [acc[-3]],
         'acc@5': [acc[-2]],
         'acc@10': [acc[-1]]}
        )
    print(acc_df)
    ax = sns.barplot(data=acc_df, ci=None)
    ax.figure.savefig("figures/barplot.png")


if __name__ == "__main__":
    with open('branch2learn/log/02_train_0326_085412.txt') as log:
        str_log = log.read()
        
            
        train_regex = re.compile('Train loss: [0-9.]{5}')
        train_loss = train_regex.findall(str_log)
        train_loss = [float(x[-5:]) for x in train_loss]

        val_regex = re.compile('Valid loss: [0-9.]{5}')
        val_loss = val_regex.findall(str_log)
        val_loss = [float(x[-5:]) for x in val_loss]

        for i in range(len(val_loss)):
            #print(i)
            print(f'({i+1}, {val_loss[i]})', end=' ')

        for i in range(len(train_loss)):
            #print(i)
            print(f'({i+2}, {train_loss[i]})', end=' ')
        