import subprocess
import sys
import os

def main():
    cmd = [sys.executable, "run_frontier.py", "qwen", "optimized"] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
