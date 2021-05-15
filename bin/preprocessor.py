import sys
import re
import tempfile

args = sys.argv[1:]
for i in range(len(args)):
    if args[i].endswith('.c'):
        source_idx = i
        source = args[i]
        break

with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as output:
    with open(source, 'r') as input:
        line = input.readline()
        while line:
            if 'TEMPLATE_START(' in line:
                # Parse _DTYPE and _TYPE
                tokens = re.split(',|\(|\)', line)
                tokens.pop(0) # drop TEMPLATE_START
                types = []
                while(len(tokens) >= 2):
                    types.append([tokens.pop(0).strip(), tokens.pop(0).strip()])

                # Parse template
                line = input.readline()
                template = []
                while 'TEMPLATE_END()' not in line:
                    if line[0] != '#':
                        template.append(line)

                    line = input.readline()

                for type_set in types:
                    dtype = type_set[0]
                    type = type_set[1]

                    for line in template:
                        line = line.replace('_DTYPE', dtype)
                        line = line.replace('_TYPE', type)

                        output.write(line)

                line = input.readline()
            else:
                output.write(line)
                line = input.readline()

    args[source_idx] = output.name

print(' '.join(args))
