FROM python:3.7
WORKDIR /app
ENV MODEL /app/model.h5
COPY requirements.txt /app
RUN python3.7 -m pip install --upgrade pip
RUN pip install -r ./requirements.txt
COPY train.py /app
COPY inference.py /app
COPY img_345.jpg /app
COPY model.h5 /app
CMD ["python", "inference.py"]~