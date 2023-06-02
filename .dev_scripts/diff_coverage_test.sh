#!/bin/bash

readarray -t IGNORED_FILES < $( dirname "$0" )/covignore.cfg
REUSE_COVERAGE_REPORT=${REUSE_COVERAGE_REPORT:-0}
REPO=${1:-"origin"}
BRANCH=${2:-"refactor_dev"}

git fetch $REPO $BRANCH

PY_FILES=""
for FILE_NAME in $(git diff --name-only ${REPO}/${BRANCH}); do
    # Only test python files in mmdet/ existing in current branch, and not ignored in covignore.cfg
    if [ ${FILE_NAME: -3} == ".py" ] && [ ${FILE_NAME:0:6} == "mmdet/" ] && [ -f "$FILE_NAME" ]; then
        IGNORED=false
        for IGNORED_FILE_NAME in "${IGNORED_FILES[@]}"; do
            # Skip blank lines
            if [ -z "$IGNORED_FILE_NAME" ]; then
                continue
            fi
            if [ "${IGNORED_FILE_NAME::1}" != "#" ] && [[ "$FILE_NAME" =~ $IGNORED_FILE_NAME ]]; then
                echo "Ignoring $FILE_NAME"
                IGNORED=true
                break
            fi
        done
        if [ "$IGNORED" = false ]; then
            PY_FILES="$PY_FILES $FILE_NAME"
        fi
    fi
done

# Only test the coverage when PY_FILES are not empty, otherwise they will test the entire project
if [ ! -z "${PY_FILES}" ]
then
    if [ "$REUSE_COVERAGE_REPORT" == "0" ]; then
        coverage run --branch --source mmdet -m pytest tests/
    fi
    coverage report --fail-under 80 -m $PY_FILES
    interrogate -v --ignore-init-method --ignore-module --ignore-nested-functions --ignore-magic --ignore-regex "__repr__" --fail-under 95 $PY_FILES
fi
