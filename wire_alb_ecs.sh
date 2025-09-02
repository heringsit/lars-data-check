set -euo pipefail

REGION=ap-northeast-2
CLUSTER=lars-data-check
SERVICE=lars-data-check-service
TG_NAME=lars-data-check-tg
ALB_NAME=lars-data-check-lb
echo ">> Discovering task/container/port..."
TASKDEF_ARN=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" --query 'services[0].taskDefinition' --output text)
CONTAINER=$(aws ecs describe-task-definition --task-definition "$TASKDEF_ARN" --region "$REGION" --query 'taskDefinition.containerDefinitions[0].name' --output text)
CONTAINER_PORT=$(aws ecs describe-task-definition --task-definition "$TASKDEF_ARN" --region "$REGION" --query 'taskDefinition.containerDefinitions[0].portMappings[0].containerPort' --output text)

echo ">> Looking up ALB/TG..."
ALB_ARN=$(aws elbv2 describe-load-balancers --names "$ALB_NAME" --region "$REGION" --query 'LoadBalancers[0].LoadBalancerArn' --output text)
ALB_DNS=$(aws elbv2 describe-load-balancers --load-balancer-arns "$ALB_ARN" --region "$REGION" --query 'LoadBalancers[0].DNSName' --output text)
ALB_SG=$(aws elbv2 describe-load-balancers --load-balancer-arns "$ALB_ARN" --region "$REGION" --query 'LoadBalancers[0].SecurityGroups[0]' --output text)
TG_ARN=$(aws elbv2 describe-target-groups --names "$TG_NAME" --region "$REGION" --query 'TargetGroups[0].TargetGroupArn' --output text)

echo "ALB=$ALB_NAME  DNS=$ALB_DNS"
echo "TG=$TG_NAME    ARN=$TG_ARN"
echo "Container=$CONTAINER  Port=$CONTAINER_PORT"

# Ensure 443 forwards to your TG and 80 redirects to 443
L80=$(aws elbv2 describe-listeners --load-balancer-arn "$ALB_ARN" --region "$REGION" --query 'Listeners[?Port==`80`].ListenerArn' --output text)
L443=$(aws elbv2 describe-listeners --load-balancer-arn "$ALB_ARN" --region "$REGION" --query 'Listeners[?Port==`443`].ListenerArn' --output text)
[ -n "$L443" ] && aws elbv2 modify-listener --listener-arn "$L443" --region "$REGION" --default-actions Type=forward,TargetGroupArn="$TG_ARN" >/dev/null
if [ -z "$L80" ] || [ "$L80" = "None" ]; then
  aws elbv2 create-listener --load-balancer-arn "$ALB_ARN" --region "$REGION" \
    --protocol HTTP --port 80 \
    --default-actions Type=redirect,RedirectConfig='{"Protocol":"HTTPS","Port":"443","StatusCode":"HTTP_301"}' >/dev/null
else
  aws elbv2 modify-listener --listener-arn "$L80" --region "$REGION" \
    --default-actions Type=redirect,RedirectConfig='{"Protocol":"HTTPS","Port":"443","StatusCode":"HTTP_301"}' >/dev/null || true
fi

# Allow ALB -> Task on 8080
TASK_SG=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" --query 'services[0].networkConfiguration.awsvpcConfiguration.securityGroups[0]' --output text)
aws ec2 authorize-security-group-ingress --group-id "$TASK_SG" --protocol tcp --port 8080 --source-group "$ALB_SG" --region "$REGION" >/dev/null || true

# Attach service to TG so targets auto-register
aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --load-balancers targetGroupArn="$TG_ARN",containerName="$CONTAINER",containerPort="$CONTAINER_PORT" \
  --force-new-deployment \
  --region "$REGION" >/dev/null

aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION"
echo ">> Service stable. Checking target health:"
aws elbv2 describe-target-health --target-group-arn "$TG_ARN" --region "$REGION" \
  --query 'TargetHealthDescriptions[].{Target:Target.Id,Port:Target.Port,State:TargetHealth.State,Reason:TargetHealth.Reason}' --output table

echo "Try: https://data.larscare.com/healthz  and  https://data.larscare.com/login"
