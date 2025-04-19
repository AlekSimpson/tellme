from tellme import tellme
import argparse
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Assistant CLI command.")

    parser.add_argument('input', type=tellme, help='Ask about a command.')
    # parser.add_argument('-n', '--number', type=int, default=10, help='A number (default: 10)')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')

    args = parser.parse_args()

    print(args.input.strip())

