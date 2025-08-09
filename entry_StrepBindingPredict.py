import traceback
from stallearn_slim import StallingRnnNet


def work():
    try:
        import StrepBindingPredict
        StrepBindingPredict.task()
    except:
        f = open("./output/output.md", "a")
        print("\n```\n", file=f, )
        traceback.print_exc(file=f)
        print("\n```\n", file=f, )
        f.close()
    # zip_all_output()


if __name__ == '__main__':
    work()
