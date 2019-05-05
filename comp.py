import plotter
import matplotlib.pyplot as plt

def plot(name, graph, color):
    accuracy, count = graph.loadGraph()
    plt.plot(count, accuracy, color)

def makeGraph(name, color='red'):
    graph = plotter.SavedAccuracy(name)
    plot(name, graph, color)

makeGraph('cifar10_dropout_acc.pkl', color='blue')
#makeGraph('svhn_tabu_acc.pkl', color='green')
makeGraph('cifar10_adaptive_acc.pkl', color='red')
#makeGraph('fmnist_non_dropout_acc.pkl', color='green')

plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.legend(['Dropout', 'Adaptive', 'Non'])
plt.show()
