import torch

class SavedAccuracy():
    
    def __init__(self, name="data.pkl"):
        self.name = name
    
    def saveGraph(self, last_acc, last_counter):
        graph = {'last_acc': last_acc, 'last_counter': last_counter}
        torch.save(graph, "./acc_data/"+ self.name) 
        
    def loadGraph(self):
        graph = torch.load("./acc_data/"+ self.name)
        last_acc_fetch = graph['last_acc']
        last_counter_fetch = graph['last_counter']
        return (last_acc_fetch, last_counter_fetch)
    
