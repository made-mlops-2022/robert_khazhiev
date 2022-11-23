import requests

url = "http://127.0.0.1:8000"


def predict() -> list:
    predict_url = url + "/predict"
    resp = requests.get(predict_url).text
    lst_resp = [int(i) for i in resp[1:-1].split(",")]
    return lst_resp


def check_health() -> bool:
    health_url = url + "/health"
    resp = requests.get(health_url).status_code
    if resp == 200:
        return True
    else:
        return False


if __name__ == "__main__":
    alive = check_health()
    if alive:
        print(predict())
    else:
        print("SERVICE UNAVAILABLE")
