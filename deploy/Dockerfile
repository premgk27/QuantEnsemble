# BTC Signal Lambda — container image
# Build: docker build -t btc-signal -f deploy/Dockerfile .
# Test:  docker run -p 9000:8080 -e S3_BUCKET=my-bucket btc-signal
#        curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'

FROM public.ecr.aws/lambda/python:3.11

# Install Python dependencies
COPY deploy/requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

# Copy source code
COPY src/btc_data.py        ${LAMBDA_TASK_ROOT}/
COPY src/btc_predict.py     ${LAMBDA_TASK_ROOT}/
COPY src/btc_trade.py       ${LAMBDA_TASK_ROOT}/
COPY src/features.py        ${LAMBDA_TASK_ROOT}/
COPY src/data_loader.py     ${LAMBDA_TASK_ROOT}/
COPY src/lambda_handler.py  ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.handler"]
