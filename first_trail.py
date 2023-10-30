import mlflow


def calculate(x,y):
    return x+y


if __name__ == "__main__":
    with mlflow.start_run():
        x,y = 22,31
        z = calculate(x,y)
        mlflow.log_param("X",x)
        mlflow.log_param("Y",y)
        mlflow.log_param("Z",z)
        print("Running scuccessfully...")
     