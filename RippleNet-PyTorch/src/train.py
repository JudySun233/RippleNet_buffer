import numpy as np
import torch
import os

from model import RippleNet


def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    model = RippleNet(args, n_entity, n_relation)
    if args.use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
    )

    # Directory to save checkpoints
    checkpoint_dir = "./checkpoints"
    results_dir = "./results"
    log_file_path = os.path.join(results_dir, "train_log.txt")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    log_file = open(log_file_path, "w")
    best_eval_auc = 0
    best_model_path = None

    for step in range(args.n_epoch):
        print(f"Epoch {step + 1}/{args.n_epoch}: Model is on device: {next(model.parameters()).device}")
        # training
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += args.batch_size
            if show_loss:
                loss_message = '%.1f%% %.4f\n' % (start / train_data.shape[0] * 100, loss.item())
                print(loss_message.strip())
                log_file.write(loss_message)

        # evaluation
        train_auc, train_acc = evaluation(args, model, train_data, ripple_set, args.batch_size)
        eval_auc, eval_acc = evaluation(args, model, eval_data, ripple_set, args.batch_size)
        # test_auc, test_acc = evaluation(args, model, test_data, ripple_set, args.batch_size)

        epoch_message = (
            f"Epoch {step + 1}    Train AUC: {train_auc:.4f}  ACC: {train_acc:.4f}    "
            f"Eval AUC: {eval_auc:.4f}  ACC: {eval_acc:.4f}\n"
        )
        print(epoch_message.strip())
        log_file.write(epoch_message)

        # Save model checkpoint for this epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{step + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

        # Update best model if necessary
        if eval_auc > best_eval_auc:
            best_eval_auc = eval_auc
            best_model_path = os.path.join(checkpoint_dir, "best_model_epoch.pt")
            torch.save(model.state_dict(), best_model_path)
            best_model_message = (
                f"New best model saved: Epoch {step + 1}, AUC: {best_eval_auc:.4f}, ACC: {eval_acc:.4f}\n"
            )
            print(best_model_message.strip())
            log_file.write(best_model_message)
            
    log_file.close()
    print(f"Training completed. Best model saved at: {best_model_path}")
    return best_model_path


def test(args, data_info, best_model_path, k=10):
    """Test the model on the test dataset and generate predictions and recommendations."""
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    model = RippleNet(args, n_entity, n_relation)
    if args.use_cuda:
        model.cuda()

    # Load the best model checkpoint
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Test phase
    test_auc, test_acc = evaluation(args, model, test_data, ripple_set, args.batch_size)
    print(f"Test Results: AUC: {test_auc:.4f}, ACC: {test_acc:.4f}")

    # Load user history and test data
    user_history_path = "../data/movie/user_history_dict.npy"
    test_data_path = "../data/movie/test_data.npy"
    user_history_dict = np.load(user_history_path, allow_pickle=True).item()
    test_data = np.load(test_data_path)

    # Directory to save results
    results_path = "./results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Output file paths
    predictions_file = os.path.join(results_path, "predictions_list.txt")
    top_k_file = os.path.join(results_path, "top_k_recommendations.txt")
    metrics_file = os.path.join(results_path, "metrics.txt")

    # Initialize
    predictions = []
    user_recommendations = {}
    ground_truth = {}
    all_items = np.arange(model.entity_emb.num_embeddings)
    hit_count = 0
    total_relevant = 0
    total_recommended = 0
    ndcg_sum = 0
    user_count = 0

    # Iterate over all users in the test data
    unique_users = np.unique(test_data[:, 0])
    for user in unique_users:
        # Step 1: Get user's interaction history and candidate items
        interacted_items = user_history_dict.get(user, [])
        candidate_items = list(set(all_items) - set(interacted_items))
        # Step 2: Prepare memories for the user
        memories_h, memories_r, memories_t = [], [], []
        for i in range(args.n_hop):
            memories_h.append(torch.LongTensor(ripple_set[user][i][0]).unsqueeze(0))
            memories_r.append(torch.LongTensor(ripple_set[user][i][1]).unsqueeze(0))
            memories_t.append(torch.LongTensor(ripple_set[user][i][2]).unsqueeze(0))

        # Prepare all items for the user
        items = torch.LongTensor(candidate_items)
        labels = torch.zeros_like(items)

        if args.use_cuda:
            items = items.cuda()
            labels = labels.cuda()
            memories_h = [m.cuda() for m in memories_h]
            memories_r = [m.cuda() for m in memories_r]
            memories_t = [m.cuda() for m in memories_t]

        # Step 3: Predict scores for all candidate items
        scores = model(items, labels, memories_h, memories_r, memories_t)["scores"].detach().cpu().numpy()
        predictions.append((user, scores.tolist()))

        # Step 4: Select Top-K items
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_items = [(candidate_items[i], scores[i]) for i in top_k_indices]
        user_recommendations[user] = top_k_items

        # Step 5: Get ground truth
        ground_truth[user] = set(test_data[test_data[:, 0] == user][:, 1][test_data[test_data[:, 0] == user][:, 2] == 1])

        # Step 6: Calculate metrics
        recommended_items = [item for item, _ in top_k_items]
        hits = ground_truth[user] & set(recommended_items)
        hit_count += len(hits)
        total_relevant += len(ground_truth[user])
        total_recommended += len(recommended_items)

        relevance_scores = [1 if item in ground_truth[user] else 0 for item in recommended_items]
        ndcg_sum += calculate_ndcg(relevance_scores)
        user_count += 1

    # Step 7: Save predictions
    with open(predictions_file, "w") as f:
        for user, scores in predictions:
            scores_str = " ".join(map(str, scores))
            f.write(f"User {user}: {scores_str}\n")
    print(f"Predictions for all items saved to {predictions_file}")

    # Step 8: Save Top-K recommendations
    with open(top_k_file, "w") as f:
        for user, top_k_items in user_recommendations.items():
            top_k_str = ", ".join([f"({item}, {score:.6f})" for item, score in top_k_items])
            f.write(f"User {user}: [{top_k_str}]\n")
    print(f"Top-K recommendations saved to {top_k_file}")

    # Step 9: Calculate and save metrics
    precision = hit_count / total_recommended if total_recommended > 0 else 0
    recall = hit_count / total_relevant if total_relevant > 0 else 0
    ndcg = ndcg_sum / user_count if user_count > 0 else 0

    with open(metrics_file, "w") as f:
        f.write(f"Precision@{k}: {precision:.6f}\n")
        f.write(f"Recall@{k}: {recall:.6f}\n")
        f.write(f"NDCG@{k}: {ndcg:.6f}\n")
    print(f"Metrics saved to {metrics_file}")

def calculate_ndcg(relevance_scores):
    """Calculate NDCG for a single user's recommendations."""
    dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])
    idcg = sum([1 / np.log2(idx + 2) for idx in range(min(sum(relevance_scores), len(relevance_scores)))])
    return dcg / idcg if idcg > 0 else 0

def get_feed_dict(args, model, data, ripple_set, start, end):
    items = torch.LongTensor(data[start:end, 1])
    labels = torch.LongTensor(data[start:end, 2])
    memories_h, memories_r, memories_t = [], [], []
    for i in range(args.n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
    if args.use_cuda:
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))
    return items, labels, memories_h, memories_r,memories_t


def evaluation(args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    model.eval()
    while start < data.shape[0]:
        auc, acc = model.evaluate(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    model.train()
    return float(np.mean(auc_list)), float(np.mean(acc_list))