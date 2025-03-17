from llava.train.train import train

if __name__ == "__main__":
    print(train.__code__.co_filename)
    
    # 方法2：使用inspect模块（更推荐）
    import inspect
    print(inspect.getfile(train))

    train()
