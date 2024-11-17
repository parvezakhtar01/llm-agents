# FastAPI Lambda Deployment Guide

This guide explains how to deploy a FastAPI application to AWS Lambda using container images.

## Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed
- Access to AWS ECR repository
- FastAPI application code ready

## 1. Dockerfile Setup

Create a `Dockerfile` in your project root:

```dockerfile
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt awslambdaric

COPY app/ ./app/
COPY .env .env.example ./

ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD [ "app.main.handler" ]
2. Configure FastAPI for Lambda
Update your app/main.py:
pythonCopyfrom fastapi import FastAPI
from mangum import Mangum
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Your API Title",
    description="Your API Description",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your routes here
app.include_router(your_router, prefix="/api/v0")

# Lambda handler
handler = Mangum(app, lifespan="off", api_gateway_base_path=None)
3. Build and Push Docker Image
bashCopy# Build image for amd64 architecture
docker build --platform=linux/amd64 -t fastapi-lambda .

# Tag image for ECR (replace with your account/region)
docker tag fastapi-lambda:latest 604982989727.dkr.ecr.ap-south-1.amazonaws.com/api/stock-analyzer:latest

# Login to ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 604982989727.dkr.ecr.ap-south-1.amazonaws.com

# Push to ECR
docker push 604982989727.dkr.ecr.ap-south-1.amazonaws.com/api/stock-analyzer:latest
4. Create Lambda Function
bashCopy# Create IAM role for Lambda
aws iam create-role \
    --role-name lambda-admin-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach necessary policy (for testing, using admin access)
aws iam attach-role-policy \
    --role-name lambda-admin-role \
    --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

# Wait for IAM role propagation
sleep 15

# Create Lambda function
aws lambda create-function \
    --region ap-south-1 \
    --function-name stock-analyzer \
    --package-type Image \
    --code ImageUri=604982989727.dkr.ecr.ap-south-1.amazonaws.com/api/stock-analyzer:latest \
    --role arn:aws:iam::604982989727:role/lambda-admin-role \
    --timeout 300 \
    --memory-size 512
5. Update Existing Lambda Function
bashCopy# Update function code (after making changes)
aws lambda update-function-code \
    --region ap-south-1 \
    --function-name stock-analyzer \
    --image-uri 604982989727.dkr.ecr.ap-south-1.amazonaws.com/api/stock-analyzer:latest
6. Test Lambda Function
Test GET endpoint
bashCopyaws lambda invoke \
    --region ap-south-1 \
    --function-name stock-analyzer \
    --cli-binary-format raw-in-base64-out \
    --payload '{
        "version": "2.0",
        "routeKey": "GET /api/v0/items",
        "rawPath": "/api/v0/items",
        "headers": {
            "accept": "application/json"
        },
        "requestContext": {
            "http": {
                "method": "GET",
                "path": "/api/v0/items",
                "sourceIp": "127.0.0.1",
                "userAgent": "aws-cli/2.0"
            }
        },
        "isBase64Encoded": false
    }' \
    response.json
Test POST endpoint
bashCopyaws lambda invoke \
    --region ap-south-1 \
    --function-name stock-analyzer \
    --cli-binary-format raw-in-base64-out \
    --payload '{
        "version": "2.0",
        "routeKey": "POST /api/v0/stocks/analyze",
        "rawPath": "/api/v0/stocks/analyze",
        "headers": {
            "accept": "application/json",
            "Content-Type": "application/json"
        },
        "requestContext": {
            "http": {
                "method": "POST",
                "path": "/api/v0/stocks/analyze",
                "sourceIp": "127.0.0.1",
                "userAgent": "aws-cli/2.0"
            }
        },
        "body": "{\"instruction\":\"Analyze tech stocks AAPL MSFT GOOG\",\"parameters\":{}}",
        "isBase64Encoded": false
    }' \
    response.json
7. View Lambda Logs
bashCopy# Watch logs in real-time
aws logs tail /aws/lambda/stock-analyzer --follow

# Or get recent logs
aws logs get-log-events \
    --region ap-south-1 \
    --log-group-name "/aws/lambda/stock-analyzer" \
    --log-stream-name=$(aws logs describe-log-streams \
        --region ap-south-1 \
        --log-group-name "/aws/lambda/stock-analyzer" \
        --order-by LastEventTime \
        --descending \
        --max-items 1 \
        --query 'logStreams[0].logStreamName' \
        --output text)
Notes

Replace account ID (604982989727), region (ap-south-1), and repository name as needed
Adjust memory and timeout settings based on your application needs
Consider using more restrictive IAM policies in production
The awslambdaric package is required for Lambda container images
Use --platform=linux/amd64 when building on ARM machines (like M1/M2 Macs)


# 1. Create API Gateway (HTTP API type)
aws apigateway create-rest-api \
    --region ap-south-1 \
    --name "stock-analyzer-api" \
    --endpoint-configuration types=REGIONAL

# Store the API ID for use in subsequent commands
API_ID=$(aws apigateway get-rest-apis \
    --query 'items[?name==`stock-analyzer-api`].id' \
    --output text)

# 2. Get the root resource ID
ROOT_RESOURCE_ID=$(aws apigateway get-resources \
    --rest-api-id $API_ID \
    --query 'items[?path==`/`].id' \
    --output text)

# 3. Create a proxy resource with {proxy+}
aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $ROOT_RESOURCE_ID \
    --path-part "{proxy+}"

# Store the proxy resource ID
PROXY_RESOURCE_ID=$(aws apigateway get-resources \
    --rest-api-id $API_ID \
    --query 'items[?path==`/{proxy+}`].id' \
    --output text)

# 4. Create ANY method on the proxy resource
aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $PROXY_RESOURCE_ID \
    --http-method ANY \
    --authorization-type NONE

# 5. Set up Lambda integration
aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $PROXY_RESOURCE_ID \
    --http-method ANY \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri arn:aws:apigateway:ap-south-1:lambda:path/2015-03-31/functions/arn:aws:lambda:ap-south-1:604982989727:function:stock-analyzer/invocations

# 6. Add Lambda permission for API Gateway
aws lambda add-permission \
    --function-name stock-analyzer \
    --statement-id apigateway-permission \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:ap-south-1:604982989727:$API_ID/*/*/*"

# 7. Deploy the API
aws apigateway create-deployment \
    --rest-api-id $API_ID \
    --stage-name v1

# 8. Get the API endpoint URL
```bash
 echo "API Endpoint: https://$API_ID.execute-api.ap-south-1.amazonaws.com/v1"
 ```

Test GET endpoint (/api/v0/items):

```bash 
curl -X GET "https://ffom4klve4.execute-api.ap-south-1.amazonaws.com/v1/api/v0/items" \
-H "accept: application/json"
```

Test POST endpoint (/api/v0/stocks/analyze):

``` bash 
curl -X POST "https://ffom4klve4.execute-api.ap-south-1.amazonaws.com/v1/api/v0/stocks/analyze" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
    "instruction": "Analyze tech stocks AAPL MSFT GOOG",
    "parameters": {}
}'
```
