import subprocess
import sys
import os
def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(base_path, "run_frontier.py"), "gemma", "baseline"] + sys.argv[1:])
if __name__ == "__main__":
    main()
