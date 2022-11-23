# VK Technopark-BMSTU | SEM II, ML OPS | HW_2

================================================================.  
Хажиев Роберт Ринатович.  
Группа ML-21.

Преподаватели: Михаил Марюфич


### Quick Start:
Build a docker image
~~~
docker build -t rkhazh/ml_ops:v0 .
~~~

(Or) Pull a docker image from docker-hub
~~~
cd online_inference
docker pull rkhazh/ml_ops:v0
~~~

### Quick Run:
~~~
cd online_inference
docker run --name online_inference -p 8000:8000 rkhazh/ml_ops:v0
~~~
Service-swagger is available now on http://127.0.0.1:8000/docs


### Fetcher Test:
~~~
python3 fetcher.py
~~~

### Pytest:
~~~
cd online_inference
pytest test.py
~~~
