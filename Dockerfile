FROM python:3.9.7-slim-bullseye

RUN apt-get update && apt-get install -y python-dev gcc && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install cython

WORKDIR /clearbox-engine

COPY . .

RUN pip install -r requirements.txt

RUN python setup.py build_ext 

RUN python setup.py bdist_wheel

RUN pip install dist/*.whl

RUN cd .. && rm -rf clearbox-engine

COPY ./requirements.txt /requirements.txt

WORKDIR /