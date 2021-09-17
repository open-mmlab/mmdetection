pipeline {
    agent {
        label 'otecheck'
    }
    stages {
        stage('Configure') {
            steps {
                script {
                    checkout scm
                    shortCommit = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
                    echo shortCommit
                }
            }
        }

        stage("Run smoke and regression tests on commit") {
            steps {
                sh """
                env
                """
                echo shortCommit
                build job: "ote/util-common/ote-test-on-commit", parameters: [[$class: 'StringParameterValue', name: 'OTECOMMIT', value: shortCommit]]
            }
        }
    }
}
