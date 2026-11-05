import matplotlib.pyplot as plt

def config():
    plt.rc('font', size=14)  # set the default fontsize for the text in plots
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.formatter.use_mathtext'] = True

config()