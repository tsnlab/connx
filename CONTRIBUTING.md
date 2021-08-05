# CONNX contribution guidelines
Welcome to CONNX project and very welcome your contirbutions.

 1. Join our connx community. [![Gitter](https://badges.gitter.im/c-onnx/community.svg)](https://gitter.im/c-onnx/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
 2. Don't hesitate to send message to maintainer(@semihlab)
 3. If you are interested in contributing code, please follow contribution guilde lines below

# How to add new operator
 1. Implement operator in src/opset directory
 2. Convert ONNX test case to CONNX using onnx-connx's bin/convert utility
 3. ports/linux$ ninja test

# How to contribute code
 1. Please agree CLA
 2. Pass all the onnx test cases
 3. Check C lint (clang-format-10 is required)

~~~sh
$ bin/check-clang-format
~~~

 4. Check python lint (flake8 is required)

~~~sh
$ flake8
~~~

 5. Register lint to git commit hook (optional)

~~~sh
$ cp bin/check-clang-format .git/hooks/pre-commit  # Register C lint only
$ ln -s /usr/bin/flake8 .git/hooks/pre-commit      # Register Python lint only
$ cp bin/lint .git/hooks/pre-commit                # Register C and Python lint at same time
~~~

 6. Pull request to maintainer(@semihlab)

