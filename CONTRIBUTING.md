# Contributing to This Repository


## Process

해당 Repository에 기여하는 방법을 소개드립니다.

### 1. Fork the Repository

이 저장소를 자신의 GitHub 계정으로 fork합니다.

### 2. Clone the Forked Repository

fork한 저장소를 로컬 디렉토리로 클론합니다.

```bash
# in your workspace
$ git clone [fork Repository URL]
$ cd [forked Repository Name]
```

### 3.Changes and Commit
로컬에서 변경 사항을 적용하고 커밋합니다.
```bash
$ git add .
$ git commit -m "Describe your changes"
$ git push origin main
```

### 4. Create a Pull Request
GitHub 웹사이트로 이동하여 fork 저장소에 Pull Request를 생성합니다.
`Pull Request`를 등록해주세요.

### Optional. Resolve Conflict

Pull Request 를 등록했는데, conflict 가 있어서 auto merge 가 안된다고 하는 경우 해당 conflict 를 해결해주세요.

```bash
# in Interview_Question_for_Beginner
$ git remote add --track main upstream https://github.com/hoon0303/Google_ML_Bootcamp_2024.git
$ git fetch upstream
$ git rebase upstream/main
# (resolve conflict in your editor)
$ git add .
$ git rebase --continue
$ git push -f origin main

