import sys
import re
import tempfile

args = sys.argv[1:]
for i in range(len(args)):
    if args[i].endswith('.c'):
        source_idx = i
        source = args[i]
        break

def is_DTYPE(dtype):
    if dtype == 'UINT8':
        return True
    elif dtype == 'INT8':
        return True
    elif dtype == 'UINT16':
        return True
    elif dtype == 'INT16':
        return True
    elif dtype == 'UINT32':
        return True
    elif dtype == 'INT32':
        return True
    elif dtype == 'UINT64':
        return True
    elif dtype == 'INT64':
        return True
    elif dtype == 'FLOAT16':
        return True
    elif dtype == 'FLOAT32':
        return True
    elif dtype == 'FLOAT64':
        return True
    elif dtype == 'STRING':
        return True
    elif dtype == 'BOOL':
        return True
    elif dtype == 'COMPLEX64':
        return True
    elif dtype == 'COMPLEX128':
        return True
    else:
        return False

def get_TYPE(dtype):
    if dtype == 'UINT8':
        return 'uint8_t'
    elif dtype == 'INT8':
        return 'int8_t'
    elif dtype == 'UINT16':
        return 'uint16_t'
    elif dtype == 'INT16':
        return 'int16_t'
    elif dtype == 'UINT32':
        return 'uint32_t'
    elif dtype == 'INT32':
        return 'int32_t'
    elif dtype == 'UINT64':
        return 'uint64_t'
    elif dtype == 'INT64':
        return 'int64_t'
    elif dtype == 'FLOAT16':
        return 'float16_t'
    elif dtype == 'FLOAT32':
        return 'float32_t'
    elif dtype == 'FLOAT64':
        return 'float64_t'
    elif dtype == 'STRING':
        return 'char*'
    elif dtype == 'BOOL':
        return 'bool'
    elif dtype == 'COMPLEX64':
        return 'void*'
    elif dtype == 'COMPLEX128':
        return 'void*'
    else:
        return '***** Not expected _DTYPE *****'

def get_NAME(dtype):
    if dtype == 'UINT8':
        return 'Uint8'
    elif dtype == 'INT8':
        return 'Int8'
    elif dtype == 'UINT16':
        return 'Uint16'
    elif dtype == 'INT16':
        return 'Int16'
    elif dtype == 'UINT32':
        return 'Uint32'
    elif dtype == 'INT32':
        return 'Int32'
    elif dtype == 'UINT64':
        return 'Uint64'
    elif dtype == 'INT64':
        return 'Int64'
    elif dtype == 'FLOAT16':
        return 'Float16'
    elif dtype == 'FLOAT32':
        return 'Float32'
    elif dtype == 'FLOAT64':
        return 'Float64'
    elif dtype == 'STRING':
        return 'String'
    elif dtype == 'BOOL':
        return 'Bool'
    elif dtype == 'COMPLEX64':
        return 'Complex64'
    elif dtype == 'COMPLEX128':
        return 'Complex128'
    else:
        return '***** Not expected _NAME *****'

with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as output:
    line_no = 1
    with open(source, 'r') as input:
        line = input.readline()
        while line:
            if 'TEMPLATE_START(' in line:
                # Parse _DTYPE and _TYPE
                tokens = re.split(',|\(|\)', line)
                tokens.pop(0) # drop TEMPLATE_START
                dtypes = []
                while(len(tokens) > 0):
                    name = tokens.pop(0).strip()
                    if is_DTYPE(name):
                        dtypes.append(name)

                # Parse template
                line = input.readline()
                template = []
                while 'TEMPLATE_END()' not in line:
                    if line[0] == '#':
                        template.append('')
                    else:
                        template.append(line)

                    line = input.readline()

                for dtype in dtypes:
                    type = get_TYPE(dtype)
                    name = get_NAME(dtype)

                    for idx, (line) in enumerate(template):
                        line = line.replace('TEMPLATE_DTYPE', dtype)
                        line = line.replace('TEMPLATE_TYPE', type)
                        line = line.replace('TEMPLATE_NAME', name)

                        output.write('#line {} "{}"\n'.format(line_no + idx + 1, source))
                        output.write(line)

                line = input.readline()
                line_no += len(template) + 2 # lines of template + header + tail
            else:
                output.write('#line {} "{}"\n'.format(line_no, source))
                output.write(line)

                line = input.readline()
                line_no += 1

    args[source_idx] = output.name

print(' '.join(args))
