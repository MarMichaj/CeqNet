def print_tree(d, depth=0, print_value=False):
    for k in d.keys():
        if isinstance(d[k], dict):
            print('  ' * depth, k)
            print_tree(d[k], depth + 1, print_value)
        else:
            if print_value:
                print('  ' * depth, k, d[k])
            else:
                print('  ' * depth, k)

def print_yellow(string):
    print('\033[93m' + string + '\033[0m')

