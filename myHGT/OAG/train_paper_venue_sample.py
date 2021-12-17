import sys
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Training GNN on Paper-Venue (Conference) classification task')

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
graph = renamed_load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))
print("load graph end -------")

train_range = {t: True for t in graph.times if t != None and t >= 2000 and t <= 2010}
# train_range = {t: True for t in graph.times if t != None and t > 2000 and t < 2005}
valid_range = {t: True for t in graph.times if t != None and t >= 2015 and t <= 2016}
test_range = {t: True for t in graph.times if t != None and t > 2016}

types = graph.get_types()
'''
    cand_list stores all the Conference, which is the classification domain.
'''
cand_list = list(graph.edge_list['venue']['paper']['PV_Conference'].keys())
'''
Use CrossEntropy (log-softmax + NLL) here, since each paper can be associated with one venue.
'''
criterion = nn.NLLLoss()


y_map = {
# '1127325140': 'NIPS', # graph.node_forward['venue']['1127325140']: 606
#          '1130985203': 'KDD', # graph.node_forward['venue']['1130985203']: 912
#          '1158167855': 'CVPR', # graph.node_forward['venue']['1158167855']: 3124
    '2232857946': 'EMBC',
    '1130451194': 'ICC',
         }

y_want_label_map = {}
for k, v in graph.node_forward['venue'].items():
    if k in y_map:
        y_want_label_map[v] = k

cand_list_map = {}

def node_classification_sample(seed, pairs, time_range, batch_size):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    np.random.seed(seed)

    keys = list(pairs.keys())
    sample_paper_list_p = []

    for key in keys:
        venue_id = pairs[key][0]
        if venue_id in y_want_label_map:
            sample_paper_list_p.append(100)
        else:
            sample_paper_list_p.append(1)

    sample_paper_list_p = [float(i)/sum(sample_paper_list_p) for i in sample_paper_list_p]
    target_ids = np.random.choice(a=keys, size=batch_size, replace=False, p=sample_paper_list_p)

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
    for x_id, target_id in enumerate(target_ids):
        ylabel[x_id] = cand_list.index(pairs[target_id][0])

        if pairs[target_id][0] in y_want_label_map:
            cand_list_map[ylabel[x_id].item()] = y_want_label_map[pairs[target_id][0]]

    x_ids = np.arange(len(target_ids)) + node_dict['paper'][0]


    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel


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
            if target_id not in train_pairs:
                train_pairs[target_id] = [source_id, _time]
        elif _time in valid_range:
            if target_id not in valid_pairs:
                valid_pairs[target_id] = [source_id, _time]
        else:
            if target_id not in test_pairs:
                test_pairs[target_id] = [source_id, _time]

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

# pool = mp.Pool(args.n_pool)
# st = time.time()
# jobs = prepare_data(pool)

print("load best model begin--------------")

paper_rep_map = {}
x_ids_label = {}

best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    for s in range(10):
        print("Step: %d" % s)

        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = node_classification_sample(randint(),
                                                                                                          train_pairs,
                                                                                                          train_range,
                                                                                                          args.batch_size)
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]

        print("paper_rep shape:")
        print(paper_rep.shape)

        paper_rep_numpy = paper_rep.cpu().numpy()
        ylabel_numpy = ylabel.cpu().numpy()

        print("cand_list_map:")
        print(cand_list_map)
        print("ylabel_numpy:")
        print(ylabel_numpy)

        for i in range(len(ylabel_numpy)):
            if ylabel_numpy[i] not in cand_list_map:
                continue
            if x_ids[i] in paper_rep_map:
                paper_rep_map[x_ids[i]] = (paper_rep_map[x_ids[i]] + paper_rep_numpy[i]) / 2
            else:
                x_ids_label[x_ids[i]] = ylabel_numpy[i]
                paper_rep_map[x_ids[i]] = paper_rep_numpy[i]

        print("paper_rep_map shape:")
        print(len(paper_rep_map))

x_final = []
y_final = []

for k in paper_rep_map:
    v = paper_rep_map[k]
    y_final.append(x_ids_label[k])
    x_final.append(v)

print("y_final:")
print(y_final)

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(x_final)
df = pd.DataFrame()
df["y"] = y_final
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]
df["label"] = [y_map[cand_list_map[i]] for i in y_final]

plt.figure()

sns_plot = sns.scatterplot(x="comp-1", y="comp-2", hue="label",
                           # palette=sns.color_palette('coolwarm', as_cmap=True),
                           # palette=sns.color_palette('coolwarm', n_colors=3),
                           palette=sns.color_palette("hls", 2),
                           data=df, s=50
                           ).set(title="2 unrelated conference")
plt.savefig('./tsne.png')
print("Done!")
