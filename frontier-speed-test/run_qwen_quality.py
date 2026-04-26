import subprocess
import sys
def main():
    subprocess.run([sys.executable, "run_frontier.py", "qwen", "quality"] + sys.argv[1:])
if __name__ == "__main__":
    main()
