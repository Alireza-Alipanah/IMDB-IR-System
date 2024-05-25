import networkx as nx


class Node:
    def __init__(self, value):
        self.value = value
        self.outgoing_edges = set()
        self.incoming_edges = set()


class LinkGraph:
    """
    Use this class to implement the required graph in link analysis.
    You are free to modify this class according to your needs.
    You can add or remove methods from it.
    """
    def __init__(self):
        self.nodes = dict()

    def add_edge(self, u_of_edge, v_of_edge):
        u_node = self.nodes[u_of_edge]
        v_node = self.nodes[v_of_edge]
        u_node.outgoing_edges.add(v_node)
        v_node.incoming_edges.add(u_node)

    def add_node(self, node_to_add):
        if node_to_add in self.nodes:
            return
        self.nodes[node_to_add] = Node(node_to_add)
        
    def get_successors_values(self, node):
        return set(i.value for i in node.outgoing_edges)
    
    def get_predecessors_values(self, node):
        return set(i.value for i in node.incoming_edges)

    def get_successors(self, node, to_include):
        return self.get_successors_values(self.nodes[node]).intersection(to_include)

    def get_predecessors(self, node, to_include):
        return self.get_predecessors_values(self.nodes[node]).intersection(to_include)
