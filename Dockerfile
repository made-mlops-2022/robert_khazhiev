FROM python:3.10-slim-bullseye
RUN python3 -m pip install --upgrade pip

COPY . ./project
#RUN ls ./project
WORKDIR /project
RUN pip3 install -r requirements.txt

EXPOSE 8000
RUN apt update && apt install make
WORKDIR /project/online_inference
CMD ["make", "up"]
