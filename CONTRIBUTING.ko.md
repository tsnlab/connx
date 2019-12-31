# 기여하기
## Pull Request Checklist
Pull Request를 보내기 전에 아래 사항을 확인하세요.
  1. 본 문서를 숙지하세요.
  2. CLA에 서명 했는지 확인하세요.
  3. Coding guideline을 따랐는지 확인하세요.
  4. Unit test를 통과 했는지 확인하세요.

## CLA (Contributor License Agreement)
Code를 기여하기 전에 CLA에 서명해야합니다. CONNX 프로젝트는 Harmony 프로젝트에서 제안하는 CLA를 따릅니다. Harmony 프로젝트는 FOSS(Free Open Source Software)에 친화적인 CLA 또는 CAA를 제안하는 프로젝트입니다[1].

CONNX 프로젝트는 GPLv3와 MIT 라이센스로만 배포하는 조건하에 CLA 기여를 받습니다. 즉, 코드를 기여할 경우 언제나 오픈소스로 공개되는 것이 보장됩니다.

# Coding guideline
## Indent
들여쓰기는 4 length tab으로 통일합니다. vim에서 ':set ts=4' 옵션과 동일합니다.

## brace
void hello() {
}

## header
#ifndef __PACKAGE_FILE_H__
#define __PACKAGE_FILE_H__
#endif /* __PACKAGE_FILE_H__ */

## include
표준 include를 먼저 하고, 가장 마지막에 사용자 정의 include를 합니다.

## structure
CONNX은 proto buffer 라이브러리에 의존합니다. 아래와 같이 통일해서 사용합니다.

typedef _package_Structure {
} package_Structure;

## Operator
Operator, Attribute, Input, Output 이름은 ONNX의 표준을 그대로 따릅니다. 예를 들어 Add의 Input은 A와 B이고, Output은 C인데 C 언어에서 변수는 소문자로 쓰는  것이 관례이지만 ONNX 표준을 준수해 Operator에서만 대문자로 표기합니다.

## etc
그 외 애매모호하고 정해져 있지 않은 coding convention은 같이 고민하며 만들어가요.

# Unit test
test/testcase 디렉토리 안에 Operator_testcase.txt 형식으로 테스트케이스를 작성합니다. 이 때 Operator와 testcase 이름은 모두 ONNX의 Operators.md의 표기법을 준수합니다.

Warning이 하나도 없어야 하고, ONNX에서 제시하는 모든 테스트케이스를 통과해야합니다.

[1] Harmony, http://www.harmonyagreements.org
