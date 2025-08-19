import flwr as fl
from flwr.server.strategy import FedAvg

def weighted_average(metrics):
    accuracies = [n * m["accuracy"] for n, m in metrics]
    examples = [n for n, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    print("Starting Flower server at 127.0.0.1:8080")
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=200),
        strategy=strategy
    )

    if history and "accuracy" in history.metrics_distributed:
        accs = [acc for _, acc in history.metrics_distributed["accuracy"]]
        print("Accuracies per round:", accs)
