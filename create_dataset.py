import copy
import pandas as pd
import os.path as osp
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import networkx as nx
from node2vec import Node2Vec
import torch
from torch_geometric.data import InMemoryDataset, Data


class Confing:
    def __init__(self):
        self.interaction_file_path = "./data/NPInter2.xlsx"
        self.folds = 5
        self.result_floder = './data/'


class Node:
    def __init__(self, serial_number: int):
        self.serial_number = serial_number
        self.name = ""
        self.interaction_list = []
        self.coding_list = [] # If the encoding type of the node more than one, you should splice coding at one dimension.


class Rna(Node):
    def __init__(self, serial_number: int):
        super(Rna, self).__init__(serial_number)


class Protein(Node):
    def __init__(self, serial_number: int):
        super(Protein, self).__init__(serial_number)


class Interaction:
    def __init__(self, interaction_object_list: list[Rna, Protein], label: int):
        self.interaction_object_list = interaction_object_list
        self.serial_number_list = [interaction_object_list[0].serial_number, interaction_object_list[1].serial_number]
        self.label = label


class MyDataset(InMemoryDataset):
    def __init__(self, root, node_list=None, interaction_list: [Interaction]=None, interaction_cannot_use_list: [Interaction]=None, transform=None,
                 pre_transform=None):
        if not osp.exists(root):
            os.makedirs(root)
        if node_list is not None:
            self.node_list = node_list
            self.interaction_list = interaction_list
            self.interaction_serial_numbers_can_use_list = [interaction.serial_number_list for interaction in interaction_list]
            self.interaction_cannot_use_list = interaction_cannot_use_list
            self.interaction_serial_numbers_cannot_use_list = [interaction.serial_number_list for interaction in interaction_cannot_use_list]
            self.serial_number_to_node_index = {node.serial_number: node_index for node_index, node in enumerate(node_list)}
            self.sum_node = 0
        super(MyDataset, self).__init__(root, transform, pre_transform, pre_filter=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def get_subgraph(self, interaction: Interaction):
        x, edge_index_list, y = [], [[], []], []
        rna, protein = interaction.interaction_object_list
        interaction_serial_numbers_all_list = []
        for interaction_serial_numbers in rna.interaction_list + protein.interaction_list:
            if interaction_serial_numbers not in self.interaction_serial_numbers_cannot_use_list:
                interaction_serial_numbers_all_list.append(interaction_serial_numbers)
        subgraph_node_list = []
        subgraph_node_list += [rna, protein]
        node_serial_number_list = []
        for interaction_serial_number_list in interaction_serial_numbers_all_list:
            rna_serial_number, proteion_serial_numer = interaction_serial_number_list
            if rna_serial_number not in node_serial_number_list:
                node_serial_number_list.append(rna_serial_number)
                rna_index = self.serial_number_to_node_index[rna_serial_number]
                subgraph_node_list.append(self.node_list[rna_index])
            if proteion_serial_numer not in node_serial_number_list:
                node_serial_number_list.append(proteion_serial_numer)
                protein_index = self.serial_number_to_node_index[proteion_serial_numer]
                subgraph_node_list.append(self.node_list[protein_index])
        for index, subgraph_node in enumerate(subgraph_node_list):
            if index in [0, 1]:
                x.append([0] + subgraph_node.coding_list)
            else:
                x.append([1] + subgraph_node.coding_list)
        node_serial_number_to_subgraph_index_dict = dict()
        node_index = 0
        for node_serial_number in node_serial_number_list:
            node_serial_number_to_subgraph_index_dict.update({node_serial_number: node_index})
            node_index += 1
        for interaction_serial_number_list in interaction_serial_numbers_all_list:
            rna_serial_number, proteion_serial_numer = interaction_serial_number_list
            edge_index_list[0].append(node_serial_number_to_subgraph_index_dict[rna_serial_number])
            edge_index_list[1].append(node_serial_number_to_subgraph_index_dict[proteion_serial_numer])
            edge_index_list[1].append(node_serial_number_to_subgraph_index_dict[rna_serial_number])
            edge_index_list[0].append(node_serial_number_to_subgraph_index_dict[proteion_serial_numer])
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(interaction.label, dtype=torch.long)
        edge_index_list = torch.tensor(edge_index_list, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index_list)
        self.sum_node += len(x)
        return data

    def process(self):
        # Read data into huge `Data` list.
        if self.node_list is not None:
            data_list = []
            interaction_num = 0
            for time, interaction in enumerate(self.interaction_list):
                data = self.get_subgraph(interaction)
                data_list.append(data)
                interaction_num += 1
                if time % 100 == 0:
                    print(f"average neighbor: {self.sum_node / interaction_num}")
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])


class InteractionFile:
    """
    This is a class of interaction file.
    """
    def __init__(self, interaction_file_path: str, folds: int, result_floder: str):
        super(InteractionFile, self).__init__()
        self.interaction_file_path = interaction_file_path
        self.folds = folds
        self.result_floder = result_floder

    def _read_interaction_file(self):
        file_content_df = pd.read_excel(io=self.interaction_file_path, header=0)
        row_name_list = file_content_df.columns.to_list()
        if row_name_list != ["RNA names", "Protein names", "Labels"]:
            print(row_name_list)
            raise Exception("The  names of excel file must be \"RNA names, Protein names, Labels\"")
        return file_content_df

    def _get_positive_interaction_set(self):
        file_content_df = self._read_interaction_file()
        rna_list, protein_list, positive_interaction_list = [], [], []
        rna_name_list, protein_name_list = [], []
        serial_num, rna, protein = 0, None, None
        for index, row in file_content_df.iterrows():
            node_rna_name, node_protein_name, label = row["RNA names"], row['Protein names'], row['Labels']
            if node_rna_name not in rna_name_list:
                rna = Rna(serial_number=serial_num)
                rna.name = node_rna_name
                rna_list.append(rna)
                rna_name_list.append(node_rna_name)
                serial_num += 1
            else:
                rna_index = rna_name_list.index(node_rna_name)
                rna = rna_list[rna_index]
            if node_protein_name not in protein_name_list:
                protein = Protein(serial_number=serial_num)
                protein.name = node_protein_name
                protein_list.append(protein)
                protein_name_list.append(node_protein_name)
                serial_num += 1
            else:
                protein_index = protein_name_list.index(node_protein_name)
                protein = protein_list[protein_index]
            positive_interaction = Interaction([rna, protein], label)
            positive_interaction_list.append(positive_interaction)
        print(f"RNA number: {len(rna_list)}, Protein number: {len(protein_list)}, "
              f"Positive interaction number: {len(positive_interaction_list)}", end=" ")
        return rna_list, protein_list, positive_interaction_list

    def _random_generate_negative_interaction_set(self, rna_list, protein_list, positive_interaction_list):
        negative_interaction_list = []
        protein_index_to_serial_num_dict, rna_index_to_serial_num_dict = {}, {}
        positive_interaction_serial_num_list, negative_interaction_serial_num_list = [], []
        for index, protein in enumerate(protein_list):
            protein_index_to_serial_num_dict[index] = protein.serial_number
        for index, rna in enumerate(rna_list):
            rna_index_to_serial_num_dict[index] = rna.serial_number
        for positive_interaction in positive_interaction_list:
            temp_positive_interaction_serial_num_list = positive_interaction.serial_number_list
            positive_interaction_serial_num_list.append(temp_positive_interaction_serial_num_list)
        max_rna_index_num, max_protein_index_num = len(rna_list), len(protein_list)
        negative_interaction_num, positive_interaction_num = 0, len(positive_interaction_serial_num_list)
        while negative_interaction_num < positive_interaction_num:
            rna_index_num_random = np.random.randint(low=0, high=max_rna_index_num)
            protein_index_num_random = np.random.randint(low=0, high=max_protein_index_num)
            rna_serial_num_ranodm, protein_serial_num_ranodm = rna_index_to_serial_num_dict[rna_index_num_random], \
                                                               protein_index_to_serial_num_dict[protein_index_num_random]
            interaction_serial_num_list_ranodm = [rna_serial_num_ranodm, protein_serial_num_ranodm]
            if rna_serial_num_ranodm != protein_serial_num_ranodm and \
                    interaction_serial_num_list_ranodm not in positive_interaction_serial_num_list and \
                    interaction_serial_num_list_ranodm not in negative_interaction_serial_num_list:
                negative_interaction_serial_num_list.append(interaction_serial_num_list_ranodm)
                negative_interaction = Interaction([rna_list[rna_index_num_random],
                                                    protein_list[protein_index_num_random]], label=0)
                negative_interaction_list.append(negative_interaction)
                negative_interaction_num += 1
        print(f"Negative interaction num: {len(negative_interaction_list)}")
        return negative_interaction_list

    def _save_name_to_serial_number(self, node_list):
        write_content = "name\tserial_number\n"
        for node in node_list:
            write_content += f"{node.name}\t{node.serial_number}\n"
        with open(self.result_floder + 'name_to_serial_number.txt', 'w', encoding='utf-8') as writer:
            writer.write(write_content)
        return None

    def _add_interaction_list_to_node(self, interaction_list: [Interaction], rna_list: [Rna], protein_list: [Protein]):
        rna_serial_number_to_index_dict = {rna.serial_number: index for index, rna in enumerate(rna_list)}
        protein_serial_number_to_index_dict = {protein.serial_number: index for index, protein in enumerate(protein_list)}
        for interaction in interaction_list:
            rna_serial_num, protein_serial_num = interaction.serial_number_list
            rna_index_num, protein_index_num = rna_serial_number_to_index_dict[rna_serial_num], \
                                               protein_serial_number_to_index_dict[protein_serial_num]
            rna_list[rna_index_num].interaction_list.append(interaction.serial_number_list) \
                if interaction.serial_number_list not in rna_list[rna_index_num].interaction_list else None
            protein_list[protein_index_num].interaction_list.append(interaction.serial_number_list) \
                if interaction.serial_number_list not in protein_list[protein_index_num].interaction_list else None
        node_list = rna_list + protein_list
        self._save_name_to_serial_number(node_list)
        return None

    def _get_label_list(self, interaction_list):
        label_list = []
        for interaction in interaction_list:
            label_list.append(interaction.label)
        return label_list

    def _get_interaction_set(self):
        rna_list, protein_list, positive_interaction_list = self._get_positive_interaction_set()
        negative_interaction_list = self._random_generate_negative_interaction_set(rna_list, protein_list, positive_interaction_list)
        interaction_list = positive_interaction_list + negative_interaction_list
        self._add_interaction_list_to_node(interaction_list, rna_list, protein_list)
        return protein_list, rna_list, positive_interaction_list, negative_interaction_list, interaction_list

    def _get_global_graph(self, interaction_list: [Interaction], node_list: [Node]):
        global_graph = nx.Graph()
        [global_graph.add_node(node.serial_number) for node in node_list]
        for interaction in interaction_list:
            global_graph.add_edge(*interaction.serial_number_list)
        return global_graph

    def _generate_and_save_training_subgraph(self, global_graph: [nx.Graph], testing_interaction_per_fold_list, fold):
        training_subgraph_save_file = self.result_floder + f'training_subgraph/fold_{fold}.edgelist'
        if not osp.exists(os.path.dirname(training_subgraph_save_file)):
            os.makedirs(os.path.dirname(training_subgraph_save_file))
        training_subgraph = copy.deepcopy(global_graph)
        [training_subgraph.remove_edge(*interaction.serial_number_list) for interaction in testing_interaction_per_fold_list]
        nx.write_edgelist(training_subgraph, training_subgraph_save_file)
        return training_subgraph

    def _add_node2vec_embedding_to_node(self, node_list, fold):
        node_serial_number_to_index = {node.serial_number: index for index, node in enumerate(node_list)}
        node2vec_file = self.result_floder + f'node2vec_embedding/fold_{fold}.node2vec.embedding'
        with open(node2vec_file, 'r', encoding='utf-8') as reader:
            file_content_list = reader.readlines()[1:]
        for single_line in file_content_list:
            line_split = single_line.split()
            node_serial_number, node_node2vec_coding = int(line_split[0]), line_split[1:]
            node_node2vec_coding = [float(coding) for coding in node_node2vec_coding]
            node_index = node_serial_number_to_index[node_serial_number]
            node = node_list[node_index]
            coding = node_node2vec_coding
            node.coding_list = coding
        return None

    def _node2vec_embedding(self, subgraph, fold):
        save_file = self.result_floder + f'node2vec_embedding/fold_{fold}.node2vec.embedding'
        if not osp.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        node2vec = Node2Vec(subgraph, dimensions=64, walk_length=1, num_walks=1, workers=os.cpu_count())
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        model.wv.save_word2vec_format(save_file)
        return None

    def split_and_save_training_and_testing_data(self):
        protein_list, rna_list, positive_interaction_list, negative_interaction_list, interaction_list = self._get_interaction_set()
        label_list = self._get_label_list(interaction_list)
        folds = self.folds
        node_list = protein_list + rna_list
        global_graph = self._get_global_graph(interaction_list, node_list)
        folds = StratifiedKFold(n_splits=folds, shuffle=True).split(X=interaction_list, y=label_list)
        for fold, (train_index, test_index) in enumerate(folds):
            training_interaction_list, testing_interaction_list = [], []
            training_interaction_per_fold_list = np.array(interaction_list)[train_index].tolist()
            testing_interaction_per_fold_list = np.array(interaction_list)[test_index].tolist()
            training_interaction_list.append(training_interaction_per_fold_list)
            testing_interaction_list.append(testing_interaction_per_fold_list)
            training_subgraph = self._generate_and_save_training_subgraph(global_graph, testing_interaction_per_fold_list, fold)
            self._node2vec_embedding(training_subgraph, fold)
            self._add_node2vec_embedding_to_node(node_list, fold)
            dataset_save_floder = self.result_floder + 'dataset/'
            MyDataset(dataset_save_floder + f'fold_{fold}_train/', node_list=node_list,
                      interaction_list=training_interaction_per_fold_list,
                      interaction_cannot_use_list=testing_interaction_per_fold_list)
            MyDataset(dataset_save_floder + f'fold_{fold}_test/', node_list=node_list,
                      interaction_list=testing_interaction_per_fold_list,
                      interaction_cannot_use_list=testing_interaction_per_fold_list)
        return protein_list, rna_list, training_interaction_list, testing_interaction_list

    def get_node_list_and_training_interaction_list_and_testing_interaction_list(self):
        protein_list, rna_list, training_interaction_list, testing_interaction_list = self.split_and_save_training_and_testing_data()
        node_list = protein_list + rna_list
        return node_list, training_interaction_list, testing_interaction_list


if __name__ == '__main__':
    """Case"""
    confing = Confing()
    interactionFile = InteractionFile(confing.interaction_file_path, confing.folds, confing.result_floder)
    node_list, training_interaction_list, testing_interaction_list = \
        interactionFile.get_node_list_and_training_interaction_list_and_testing_interaction_list()