FROM python:3.7
RUN pip install torch torchvision numpy gym git+https://github.com/njustesen/ffai
COPY . /app

ENTRYPOINT ["/app/run_cart_pole.py"]