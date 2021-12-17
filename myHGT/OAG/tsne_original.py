import sys
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings
from sklearn.manifold import TSNE
import seaborn as sns
from numpy import reshape
import matplotlib.pyplot as plt
import random as random
from collections import Counter

filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Training GNN on Paper-Venue (Journal) classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./dataset/oag_output',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='PV',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_CS',
                    help='CS, Medicion or All: _CS or _Med or (empty)')
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=200,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=4,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--repeat', type=int, default=2,
                    help='How many time to train over a singe batch (reuse data)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping')

args = parser.parse_args()

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

print("load graph begin -------")
graph = renamed_load(open(os.path.join(args.data_dir, 'graph_3AI_conference%s.pk' % args.domain), 'rb'))
print("load graph end -------")

train_range = {t: True for t in graph.times if t != None}
valid_range = {t: True for t in graph.times if t != None and t >= 2015 and t <= 2016}
test_range = {t: True for t in graph.times if t != None}

types = graph.get_types()
'''
    cand_list stores all the Conference, which is the classification domain.
'''
cand_list = list(graph.edge_list['venue']['paper']['PV_Conference'].keys())
'''
Use CrossEntropy (log-softmax + NLL) here, since each paper can be associated with one venue.
'''
criterion = nn.NLLLoss()


def node_classification_sample(seed, pairs, time_range, batch_size):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    np.random.seed(seed)
    target_ids = list(pairs.keys())
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, times, edge_list, _, _ = sample_subgraph(graph, time_range, \
                                                      inp={'paper': np.array(target_info)}, \
                                                      sampled_depth=args.sample_depth, sampled_number=args.sample_width)

    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (Conference)
    '''
    masked_edge_list = []
    for i in edge_list['paper']['venue']['rev_PV_Conference']:
        if i[0] >= batch_size:
            masked_edge_list += [i]
    edge_list['paper']['venue']['rev_PV_Conference'] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['venue']['paper']['PV_Conference']:
        if i[1] >= batch_size:
            masked_edge_list += [i]
    edge_list['venue']['paper']['PV_Conference'] = masked_edge_list

    '''
        (4) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, times, edge_list, graph)
    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    ylabel = torch.zeros(len(target_ids), dtype=torch.long)
    time_array = torch.zeros(len(target_ids), dtype=torch.long)
    for x_id, target_id in enumerate(target_ids):
        ylabel[x_id] = cand_list.index(pairs[target_id][0])
        time_array[x_id] = pairs[target_id][1]
    x_ids = np.arange(len(target_ids)) + node_dict['paper'][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, time_array


def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), \
                                                               sel_train_pairs, train_range, args.batch_size))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), \
                                                           sel_valid_pairs, valid_range, args.batch_size))
    jobs.append(p)
    return jobs


train_pairs = {}
valid_pairs = {}
test_pairs = {}
'''
    Prepare all the souce nodes (Conference) associated with each target node (paper) as dict
'''

print("Prepare all the souce nodes (Conference) associated with each target node (paper) as dict begin-------")

for target_id in graph.edge_list['paper']['venue']['rev_PV_Conference']:
    for source_id in graph.edge_list['paper']['venue']['rev_PV_Conference'][target_id]:
        _time = graph.edge_list['paper']['venue']['rev_PV_Conference'][target_id][source_id]
        if _time in train_range:
            train_pairs[target_id] = [source_id, _time]

print("Prepare all the souce nodes (Conference) associated with each target node (paper) as dict end-------")

np.random.seed(43)
'''
    Only train and valid with a certain percentage of data, if necessary.
'''
sel_train_pairs = {p: train_pairs[p] for p in
                   np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage),
                                    replace=False)}
sel_valid_pairs = {p: valid_pairs[p] for p in
                   np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage),
                                    replace=False)}

stats = []
res = []
best_val = 0
train_step = 1500

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)

print("load best model begin--------------")

best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, time_array = node_classification_sample(
        randint(),
        train_pairs,
        train_range,
        args.batch_size)

    print("x_ids.shape: ")
    print(x_ids.shape)
    print("ylabel.shape: ")
    print(ylabel.shape)
    paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                            edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]

    print(paper_rep.shape)
    x_select = paper_rep.cpu().numpy()
    y_select = ylabel.cpu().numpy()
    print("y_select: ")
    print(y_select.shape)

    y_map = {
        # '1127325140': 'NIPS',
        # '1130985203': 'KDD',
        '1158167855': 'CVPR',
        # '2232857946': 'EMBC',
        # '1130451194': 'ICC',
        # '1121227772': 'ICASSP',
        '1129324708': 'MICCAI',
        '1164975091': 'ICCV',
        # '2584161585': 'ICLR',
        # '1180662882': 'ICML',
    }
    y_want_label_map = {}
    for k, v in graph.node_forward['venue'].items():
        if k in y_map:
            y_want_label_map[v] = k

    x_final = []
    y_final = []

    for i in range(len(y_select)):
        if y_select[i] in y_want_label_map:
            x_final.append(x_select[i])
            y_final.append(y_select[i])

    print("y_final: ")
    print(y_final)

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x_final)
    df = pd.DataFrame()
    df_y = []
    df_comp1 = []
    df_comp2 = []
    for i in range(len(y_final)):
        if 2010 <= time_array[i] <= 2015:
            df_y.append(y_final[i])
            df_comp1.append(z[i, 0])
            df_comp2.append(z[i, 1])

    df["y"] = df_y
    df["comp-1"] = df_comp1
    df["comp-2"] = df_comp2

    # df["y"] = y_final
    # df["comp-1"] = z[:, 0]
    # df["comp-2"] = z[:, 1]
    y_count = {}
    for y_label in df_y:
        if y_label in y_count:
            y_count[y_label] = y_count[y_label] + 1
        else:
            y_count[y_label] = 1

    # df["label"] = [(y_map[y_want_label_map[i]] + "(" + str(y_count[i]) + ")") for i in df_y]
    df["label"] = [(str(i) + ":" + y_map[y_want_label_map[i]]) for i in df_y]
    plt.figure()

    # chars = '0123456789ABCDEF'
    # n_color = df['y'].unique().shape[0]
    # n_color = len(df_y)
    # palette = ['#' + ''.join(random.sample(chars, 6)) for i in range(n_color)]
    palette = sns.color_palette("hls", 3)
    # sns_plot = sns.scatterplot(x="comp-1", y="comp-2", hue="label",
    #                            palette=sns.color_palette("hls", 3),
    #                            data=df, s=10).set(title="3 AI Conference 2010-2015")
    # lines = []
    for i, label in enumerate(df["label"].unique()):
        # add data points
        plt.scatter(x=df.loc[df['label'] == label, 'comp-1'],
                    y=df.loc[df['label'] == label, 'comp-2'],
                    color=palette[i], alpha=1, label=label,
                    s=10)

        # add label
        plt.annotate(label[:label.index(':')],
                     df.loc[df['label'] == label, ['comp-1', 'comp-2']].mean(),
                     horizontalalignment='left',
                     verticalalignment='bottom',
                     size=10, weight='bold',
                     color='white',
                     backgroundcolor=palette[i])

    # labels = [l.get_label() for l in lines]
    # plt.legend(lines, labels)
    plt.legend(loc='upper right')
    plt.savefig('./tsne.png')
    print("Done")
