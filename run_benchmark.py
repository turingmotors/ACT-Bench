from transformers import HfArgumentParser

from act_bench import ActBenchConfig, compute_score

if __name__ == "__main__":
    parser = HfArgumentParser(ActBenchConfig)
    config, *_ = parser.parse_args_into_dataclasses()
    results = compute_score(config)
    print(f"Accuracy: {results.accuracy*100:.2f}%")
    print(f"Mean ADE: {results.ade:.4f}, Mean FDE: {results.fde:.4f}")
