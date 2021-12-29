#!/usr/bin/env python3

import io
import os
import re
import sys

import jinja2

if len(sys.argv) != 3:
    print('Usage: {} [input source] [output source]'.format(sys.argv[0]))
    sys.exit(0)

input_source = os.path.abspath(sys.argv[1])
output_source = sys.argv[2]


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
        return 'complex64_t'
    elif dtype == 'COMPLEX128':
        return 'complex128_t'
    else:
        raise Exception('Not expected dtype')


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
        raise Exception('Not expected dtype')


def loop_types(*args):
    for dtype in args:
        if is_DTYPE(dtype):
            yield (dtype, get_TYPE(dtype))
        else:
            raise Exception('Not expected dtype')


consts = {
    'UINT8': 'UINT8',
    'INT8': 'INT8',
    'UINT16': 'UINT16',
    'INT16': 'INT16',
    'UINT32': 'UINT32',
    'INT32': 'INT32',
    'UINT64': 'UINT64',
    'INT64': 'INT64',
    'FLOAT16': 'FLOAT16',
    'FLOAT32': 'FLOAT32',
    'FLOAT64': 'FLOAT64',
    'STRING': 'STRING',
    'BOOL': 'BOOL',
    'COMPLEX64': 'COMPLEX64',
    'COMPLEX128': 'COMPLEX128',
    'loop_types': loop_types,
}

with open(output_source, 'w') as output:
    buffer = io.StringIO()
    line_no = 1
    buffer.write('#line {} "{}"\n'.format(line_no, input_source))

    jinja_start_tokens = ['{%', '{{', '{#']
    jinja_end_tokens = ['%}', '}}', '#}']

    with open(input_source, 'r') as input:

        line = input.readline()
        while line:
            if any(token in line for token in jinja_end_tokens):
                line += '#line {} "{}"\n'.format(line_no+1, input_source)
            elif any(token in line for token in jinja_start_tokens):
                buffer.write('#line {} "{}"\n'.format(line_no, input_source))

            if 'TEMPLATE_START(' in line:
                # Parse _DTYPE and _TYPE
                tokens = re.split(r',|\(|\)', line)
                tokens.pop(0)  # drop TEMPLATE_START
                dtypes = []
                while(len(tokens) > 0):
                    name = tokens.pop(0).strip()
                    if is_DTYPE(name):
                        dtypes.append(name)

                # Parse template
                line = input.readline()
                template = []
                while 'TEMPLATE_END()' not in line:
                    if line[0] == '#' and 'CONNX(alive)' not in line:
                        template.append('\n')
                    else:
                        template.append(line)

                    line = input.readline()

                for dtype in dtypes:
                    type = get_TYPE(dtype)
                    name = get_NAME(dtype)

                    buffer.write('#line {} "{}"\n'.format(line_no + 1, input_source))  # plus header

                    for idx, (line) in enumerate(template):
                        line = line.replace('TEMPLATE_DTYPE', 'CONNX_' + dtype)
                        line = line.replace('TEMPLATE_TYPE', type)
                        line = line.replace('TEMPLATE_NAME', name)

                        buffer.write(line)

                line = input.readline()
                line_no += len(template) + 2  # lines of template + header + tail
            else:
                buffer.write(line)

                line = input.readline()
                line_no += 1

    content = buffer.getvalue()
    rendered = jinja2.Template(content).render(
        **consts
    )
    output.write(rendered)
