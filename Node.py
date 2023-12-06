
class Node:
    def __init__(self, data=None, children={}) -> None:
        self.data = data
        self.children = children
        return
    def __repr__(self) -> str:
        return f"{self.data}: --> {self.children}"
    
    def get(self, item):
        return self.children[item]
    def set(self, data):
        self.data = data
    
    def append(self, item):
        newchild = Node(data = item, children={})
        self.children[item] = newchild