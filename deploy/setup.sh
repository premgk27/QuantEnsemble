#!/usr/bin/env bash
# AWS setup script for BTC Signal Lambda
# Run once to create all infrastructure.
# Prerequisites: aws CLI configured, Docker running, ECR + Lambda access.
#
# Usage:
#   chmod +x deploy/setup.sh
#   AWS_REGION=us-east-1 S3_BUCKET=my-btc-signals ./deploy/setup.sh

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-btc-signal-$(aws sts get-caller-identity --query Account --output text)}"
SNS_TOPIC_NAME="btc-signal-notify"
LAMBDA_FUNCTION_NAME="btc-signal-predict"
LAMBDA_ROLE_NAME="btc-signal-lambda-role"
ECR_REPO_NAME="btc-signal"
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "=== BTC Signal Lambda Setup ==="
echo "Region:   $AWS_REGION"
echo "Account:  $AWS_ACCOUNT"
echo "S3:       s3://$S3_BUCKET"
echo ""

# ---------------------------------------------------------------------------
# 1. S3 bucket
# ---------------------------------------------------------------------------
echo "[1/6] Creating S3 bucket: $S3_BUCKET"
if aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
    echo "  Bucket already exists."
else
    if [ "$AWS_REGION" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION"
    else
        aws s3api create-bucket \
            --bucket "$S3_BUCKET" \
            --region "$AWS_REGION" \
            --create-bucket-configuration LocationConstraint="$AWS_REGION"
    fi
    # Block public access
    aws s3api put-public-access-block \
        --bucket "$S3_BUCKET" \
        --public-access-block-configuration \
          "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
    echo "  Created."
fi

# ---------------------------------------------------------------------------
# 2. SNS topic for email notifications
# ---------------------------------------------------------------------------
echo "[2/6] Creating SNS topic: $SNS_TOPIC_NAME"
SNS_TOPIC_ARN=$(aws sns create-topic \
    --name "$SNS_TOPIC_NAME" \
    --region "$AWS_REGION" \
    --query TopicArn --output text)
echo "  ARN: $SNS_TOPIC_ARN"
echo ""
echo "  >>> Subscribe your email: run this command, then confirm the email <<<<"
echo "  aws sns subscribe --topic-arn $SNS_TOPIC_ARN --protocol email --notification-endpoint YOUR@EMAIL.COM"
echo ""

# ---------------------------------------------------------------------------
# 3. IAM role for Lambda
# ---------------------------------------------------------------------------
echo "[3/6] Creating IAM role: $LAMBDA_ROLE_NAME"

TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

if aws iam get-role --role-name "$LAMBDA_ROLE_NAME" 2>/dev/null; then
    echo "  Role already exists."
    ROLE_ARN=$(aws iam get-role --role-name "$LAMBDA_ROLE_NAME" --query Role.Arn --output text)
else
    ROLE_ARN=$(aws iam create-role \
        --role-name "$LAMBDA_ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --query Role.Arn --output text)
    aws iam attach-role-policy \
        --role-name "$LAMBDA_ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    echo "  Created: $ROLE_ARN"
fi

# Inline policy: S3 read/write + SNS publish
PERMISSIONS_POLICY="{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {
      \"Effect\": \"Allow\",
      \"Action\": [\"s3:GetObject\", \"s3:PutObject\"],
      \"Resource\": \"arn:aws:s3:::${S3_BUCKET}/*\"
    },
    {
      \"Effect\": \"Allow\",
      \"Action\": \"sns:Publish\",
      \"Resource\": \"${SNS_TOPIC_ARN}\"
    }
  ]
}"

aws iam put-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-name "btc-signal-permissions" \
    --policy-document "$PERMISSIONS_POLICY"
echo "  Permissions attached."

# ---------------------------------------------------------------------------
# 4. Build and push Docker image to ECR
# ---------------------------------------------------------------------------
echo "[4/6] Building and pushing Docker image to ECR..."

aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Create ECR repo if it doesn't exist
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" 2>/dev/null \
    || aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION"

# Build from project root
docker build \
    --platform linux/amd64 \
    -t "${ECR_REPO_NAME}:latest" \
    -f deploy/Dockerfile \
    .

docker tag "${ECR_REPO_NAME}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
echo "  Pushed: ${ECR_URI}:latest"

# ---------------------------------------------------------------------------
# 5. Create Lambda function
# ---------------------------------------------------------------------------
echo "[5/6] Creating Lambda function: $LAMBDA_FUNCTION_NAME"
sleep 10  # let IAM role propagate

ENV_VARS="Variables={S3_BUCKET=${S3_BUCKET},S3_MODEL_KEY=models/btc_daily_rf.pkl,SNS_TOPIC_ARN=${SNS_TOPIC_ARN}}"

if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$AWS_REGION" 2>/dev/null; then
    echo "  Updating existing function..."
    aws lambda update-function-code \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --image-uri "${ECR_URI}:latest" \
        --region "$AWS_REGION"
    aws lambda update-function-configuration \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --timeout 120 \
        --memory-size 512 \
        --environment "$ENV_VARS" \
        --region "$AWS_REGION"
else
    aws lambda create-function \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --package-type Image \
        --code ImageUri="${ECR_URI}:latest" \
        --role "$ROLE_ARN" \
        --timeout 120 \
        --memory-size 512 \
        --environment "$ENV_VARS" \
        --region "$AWS_REGION"
    echo "  Created."
fi

# ---------------------------------------------------------------------------
# 6. EventBridge schedule — daily at 00:05 UTC
# ---------------------------------------------------------------------------
echo "[6/6] Creating EventBridge schedule: daily 00:05 UTC"

RULE_ARN=$(aws events put-rule \
    --name "btc-signal-daily" \
    --schedule-expression "cron(5 0 * * ? *)" \
    --state ENABLED \
    --region "$AWS_REGION" \
    --query RuleArn --output text)

LAMBDA_ARN=$(aws lambda get-function \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --region "$AWS_REGION" \
    --query Configuration.FunctionArn --output text)

# Allow EventBridge to invoke Lambda
aws lambda add-permission \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --statement-id "allow-eventbridge" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "$RULE_ARN" \
    --region "$AWS_REGION" 2>/dev/null || true

aws events put-targets \
    --rule "btc-signal-daily" \
    --targets "Id=btc-signal-predict,Arn=${LAMBDA_ARN}" \
    --region "$AWS_REGION"

echo "  Scheduled."

# ---------------------------------------------------------------------------
# Done — summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Subscribe your email to SNS:"
echo "     aws sns subscribe --topic-arn $SNS_TOPIC_ARN \\"
echo "         --protocol email --notification-endpoint YOUR@EMAIL.COM"
echo ""
echo "  2. Upload your trained model to S3:"
echo "     uv run python src/btc_train.py --upload-s3 $S3_BUCKET"
echo ""
echo "  3. Test the Lambda manually:"
echo "     aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME \\"
echo "         --region $AWS_REGION --payload '{}' /tmp/response.json"
echo "     cat /tmp/response.json"
echo ""
echo "  Lambda will then run daily at 00:05 UTC and email you the signal."
echo ""
echo "Monthly cost estimate (Lambda free tier covers this):"
echo "  Lambda:    30 invocations/month × 512MB × ~5s = \$0.00 (free tier)"
echo "  ECR:       ~250MB image = ~\$0.03/month"
echo "  SNS email: 30 emails/month = \$0.00 (first 1,000 free)"
echo "  TOTAL:     ~\$0.03/month"
