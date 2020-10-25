# funA 作为装饰器函数
def funA(fn):
    print("C语言中文网")
    fn()  # 执行传入的fn参数
    print("http://c.biancheng.net")
    return "装饰器函数的返回值"


@funA
def funB():
    print("学习 Python")

funB
