class Node:
    def __init__(self, data={}, datatype=None) -> None:
        self.data = data
        self.datatype = datatype
        return
    def __repr__(self) -> str:
        return repr(self.data)