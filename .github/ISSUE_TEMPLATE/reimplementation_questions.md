---
name: Reimplementation Questions
about: Ask about questions during model reimplementation
title: ''
labels: 'reimplementation'
assignees: ''

---

**Notice**

There are several common situations in the reimplementation issues as below
1. Reimplement a model in the model zoo using the provided configs
2. Reimplement a model in the model zoo on other dataset (e.g., custom datasets)
3. Reimplement a custom model but all the components are implemented in MMDetection
4. Reimplement a custom model with new modules implemented by yourself

There are several things to do for different cases as below.
- For case 1 & 3, please follow the steps in the following sections thus we could help to quick identify the issue.
- For case 2 & 4, please understand that we are not able to do much help here because we usually do not know the full code and the users should be responsible to the code they write.
- One suggestion for case 2 & 4 is that the users should first check whether the bug lies in the self-implemented code or the original code. For example, users can first make sure that the same model runs well on supported datasets. If you still need help, please describe what you have done and what you obtain in the issue, and follow the steps in the following sections and try as clear as possible so that we can better help you.

**Checklist**
1. I have searched related issues but cannot get the expected help.
2. The issue has not been fixed in the latest version.

**Describe the issue**

A clear and concise description of what the problem you meet and what have you done.

**Reproduction**
1. What command or script did you run?
```
A placeholder for the command.
```
2. What config dir you run?
```
A placeholder for the config.
```
3. Did you make any modifications on the code or config? Did you understand what you have modified?
4. What dataset did you use?

**Environment**

1. Please run `python mmdet/utils/collect_env.py` to collect necessary environment information and paste it here.
2. You may add addition that may be helpful for locating the problem, such as
    - How you installed PyTorch [e.g., pip, conda, source]
    - Other environment variables that may be related (such as `$PATH`, `$LD_LIBRARY_PATH`, `$PYTHONPATH`, etc.)

**Results**

If applicable, paste the related results here, e.g., what you expect and what you get.
```
A placeholder for results comparison
```

**Issue fix**

If you have already identified the reason, you can provide the information here. If you are willing to create a PR to fix it, please also leave a comment here and that would be much appreciated!
