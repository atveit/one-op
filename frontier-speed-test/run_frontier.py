import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Frontier EML/Tropical SLC Master CLI")
    parser.add_argument("model", choices=["qwen", "gemma"], help="Model to run")
    parser.add_argument("mode", choices=["baseline", "optimized", "quality"], help="Run mode")
    parser.add_argument("--prompt", type=str, default="Explain why mathematical unification is important for hardware efficiency.")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    model_map = {
        "qwen": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "gemma": "mlx-community/gemma-4-31b-it-4bit"
    }
    
    model_path = model_map[args.model]
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    if args.mode == "quality":
        script = os.path.join(base_path, "quality_check.py")
        cmd = [sys.executable, script, "--model", model_path]
    else:
        script_name = "run_baseline.py" if args.mode == "baseline" else "run_optimized.py"
        script = os.path.join(base_path, script_name)
        cmd = [
            sys.executable, script,
            "--model", model_path,
            "--prompt", args.prompt,
            "--max-tokens", str(args.max_tokens)
        ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
