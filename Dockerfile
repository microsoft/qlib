FROM python:3.8-buster

RUN apt-get update && apt-get install -y git psmisc zip gcc g++
COPY . /qlib
RUN  pip install numpy && pip install --upgrade cython && pip install notebook \
   && cd /qlib/ && pip install . 
RUN cd /tmp && wget https://github.com/chenditc/investment_data/releases/download/2022-09-05/qlib_bin.tar.gz \
   && mkdir -p ~/.qlib/qlib_data/cn_data && tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2 \
   && rm -f qlib_bin.tar.gz
ENV JUPYTER_TOKEN=qlib
WORKDIR /qlib
CMD [ "jupyter", "notebook", "--allow-root", "--ip", "'*'"]
