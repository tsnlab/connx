import pathlib
import shutil
import subprocess

ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
SRC_PATH = ROOT_PATH / 'ports/linux'

subprocess.run('cmake . -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release', cwd=SRC_PATH, shell=True)
subprocess.run('cmake --build build', cwd=SRC_PATH, shell=True)

# Remove this part and use cmake install prefix
shutil.copy(
    SRC_PATH / 'build/libconnx.so',
    ROOT_PATH / 'pyconnx/connx/libconnx.so')
