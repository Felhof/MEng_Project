from dataclasses import dataclass


nodes = {}
SCC_lst = []


class DLLst:
    def __init__(self, items):
        self.length = len(items)
        if len(items) == 0:
            self.start = None
            self.end = None
            return
        item = items[0]
        self.start = Node(item)
        prev_node = self.start
        nodes[item] = self.start
        for item in items[1:-1]:
            next_node = Node(item)
            prev_node.next = next_node
            next_node.previous = prev_node
            nodes[item] = next_node
            prev_node = next_node
        item = items[-1]
        self.end = Node(item)
        if self.end != prev_node:
            self.end.previous = prev_node
            prev_node.next = self.end
            nodes[item] = self.end

    def __iter__(self):
        return DLLstIterator(self)


class DLLstIterator:
    def __init__(self, dllst):
        self.dllst = dllst
        self.current_node = dllst.start

    def __next__(self):
        result = self.current_node
        if result is None:
            raise StopIteration
        self.current_node = self.current_node.next
        return result


class Node:
    def __init__(self, num):
        self.num = num
        self.idx = 0
        self.level = -1
        self.low = 0
        self.next = None
        self.previous = None


class SCC:
    def __init__(self, states_dllst, lvl):
        self.states_dll = states_dllst
        self.lvl = lvl


def move_node_to_end(dllst, node):
    node.previous.next = node.next
    node.next.previous = node.previous
    dllst.end.next = node
    node.previous = dllst.end


def move_class_to_scc_lst(dllst, node, lvl):
    scc_dllst = DLLst([])
    scc_dllst.start = nodes[node.num]
    scc_dllst.end = dllst.end
    dllst.end = scc_dllst.start.previous
    scc = SCC(scc_dllst, lvl)
    SCC_lst.append(scc)
    return scc


def mta_for_scc_and_levels(dllst, mdp):
    global index
    index = 1

    def dfs_levels(current_node):
        current_node.low = index
        current_node.idx = index
        index += 1
        current_level = 0
        successors = mdp.get_successors(current_node.num)
        successor_nodes = [nodes[s] for s in successors]
        for successor_node in successor_nodes:
            if successor_node.idx == 0:
                move_node_to_end(dllst, successor_node)
                successor_level = dfs_levels(successor_node)
                current_node.low = min(current_node.low, successor_node.low)
                current_level = min(current_level, successor_level)
            elif successor_node.level == -1:
                current_node.low = min(current_node.low, successor_node.idx)
            else:
                current_level = max(current_level, successor_node.level + 1)
        if current_node.low == current_node.idx:
            scc = move_class_to_scc_lst(dllst, current_node, current_level)
            for state_node in scc.states_dll:
                state_node.level = current_level
            return current_level + 1
        return current_level

    while dllst.end is not None:
        index = 1
        current_node = dllst.end
        dfs_levels(current_node)



