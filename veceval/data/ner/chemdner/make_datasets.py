import sys


def main():
    input_prefix, output_prefix, window_size = sys.argv[1:]
    window_size = int(window_size)

    print(input_prefix, output_prefix, window_size)


if __name__ == '__main__':
    main()
