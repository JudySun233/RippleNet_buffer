import collections
import os
import numpy as np


def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    train_file = f'../data/{args.dataset}/train_data.npy'
    eval_file = f'../data/{args.dataset}/validation_data.npy'
    test_file = f'../data/{args.dataset}/test_data.npy'
    user_history_file = f'../data/{args.dataset}/user_history_dict.npy'

    # Load or split the dataset
    if os.path.exists(train_file) and os.path.exists(eval_file) and os.path.exists(test_file):
        print('Loading existing dataset splits...')
        train_data = np.load(train_file)
        eval_data = np.load(eval_file)
        test_data = np.load(test_file)
        user_history_dict = np.load(user_history_file, allow_pickle=True).item()
    else:
        print('Generating new dataset splits...')
        if os.path.exists(rating_file + '.npy'):
            rating_np = np.load(rating_file + '.npy')
        else:
            rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
            np.save(rating_file + '.npy', rating_np)

        train_data, eval_data, test_data, user_history_dict = dataset_split(rating_np, train_file, eval_file, test_file, user_history_file)

    return train_data, eval_data, test_data, user_history_dict


def dataset_split(rating_np, train_file, eval_file, test_file, user_history_file, seed=42):
    np.random.seed(seed)
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    # Save splits to .npy files
    print('Saving dataset splits to files...')
    np.save(train_file, train_data)
    np.save(eval_file, eval_data)
    if not os.path.exists(test_file):  # Ensure test_data.npy is only generated once
        np.save(test_file, test_data)
    np.save(user_history_file, user_history_dict)

    return train_data, eval_data, test_data, user_history_dict


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def get_ripple_set(args, kg, user_history_dict):
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set_file = f"../data/{args.dataset}/ripple_set.npy"

    if os.path.exists(ripple_set_file):
        print('Loading existing ripple set...')
        ripple_set = np.load(ripple_set_file, allow_pickle=True).item()
    else:
        print('Constructing ripple set...')
        ripple_set = collections.defaultdict(list)

        for user in user_history_dict:
            for h in range(args.n_hop):
                memories_h = []
                memories_r = []
                memories_t = []

                if h == 0:
                    tails_of_last_hop = user_history_dict[user]
                else:
                    tails_of_last_hop = ripple_set[user][-1][2]

                for entity in tails_of_last_hop:
                    for tail_and_relation in kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

                # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
                # this won't happen for h = 0, because only the items that appear in the KG have been selected
                # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
                if len(memories_h) == 0:
                    ripple_set[user].append(ripple_set[user][-1])
                else:
                    # sample a fixed-size 1-hop memory for each user
                    replace = len(memories_h) < args.n_memory
                    indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    ripple_set[user].append((memories_h, memories_r, memories_t))

        np.save(ripple_set_file, ripple_set)

    return ripple_set
