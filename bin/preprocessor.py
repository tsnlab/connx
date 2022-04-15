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

IS_TEMPLATE = os.environ.get('TEMPLATE', '0') == '1'


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


def pointer(arg):
    return f'{arg}*'


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

jinja2_filters = {
    'pointer': pointer,
    'to_name': get_NAME,
}

templates_dir = os.path.join(
    os.path.dirname(output_source),
    '..',
    'templates',
)

if IS_TEMPLATE and not os.path.exists(templates_dir):
    # Create templates directory
    try:
        os.makedirs(templates_dir)
    except OSError:
        pass

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(templates_dir),
    block_start_string='/*{%',
    block_end_string='%}*/',
    variable_start_string='{{',
    variable_end_string='}}',
)

jinja_env.globals.update(consts)
jinja_env.filters.update(jinja2_filters)

with open(output_source, 'w') as output:
    buffer = io.StringIO()
    buffer.write('#line {} "{}"\n'.format(1, input_source))

    jinja_start_tokens = [
        getattr(jinja_env, attr)
        for attr in ('block_start_string', 'variable_start_string', 'comment_start_string',)]
    jinja_end_tokens = [
        getattr(jinja_env, attr)
        for attr in ('block_end_string', 'variable_end_string', 'comment_end_string',)]

    with open(input_source, 'r') as input_:
        for idx, line in enumerate(input_.readlines()):
            line_no = idx + 1

            if any(token in line for token in jinja_end_tokens):
                line += '#line {} "{}"\n'.format(line_no+1, input_source)
            elif any(token in line for token in jinja_start_tokens):
                buffer.write('#line {} "{}"\n'.format(line_no, input_source))

            buffer.write(line)

    kwargs = {}

    # get op_version
    m = re.match(r'.*\/opset\/.*_([\d]+)\.c$', input_source)
    if m is not None:
        kwargs['op_version'] = int(m[1])

    content = buffer.getvalue()

    if not IS_TEMPLATE:
        rendered = jinja_env.from_string(content).render(kwargs)
    else:
        rendered = content

    output.write(rendered)
