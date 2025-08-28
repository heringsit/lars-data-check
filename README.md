# AWS Deployment (ECS Fargate + Aurora PostgreSQL)

This bundle gives you a production‑ready path to deploy the Flask app in `patient_app_csv_testing/` to AWS ECS Fargate behind an Application Load Balancer (ALB), connected to an Aurora PostgreSQL cluster.

---
## 0) Prereqs
- AWS account & CLI configured (`aws configure`).
- Docker installed.
- (Recommended) A dedicated VPC with **public subnets** (for ALB) and **private subnets** (for ECS tasks & Aurora).
- Create an **AWS Systems Manager Parameter** or **Secrets Manager** secret for DB password.

---
## 1) Build & Push Image to ECR
```bash
APP_NAME=lars-data-check
REGION=ap-northeast-2
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws ecr create-repository --repository-name ${APP_NAME} --region ${REGION} || true
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build from this bundle's root (where Dockerfile lives)
docker build -t ${APP_NAME}:latest .
docker tag ${APP_NAME}:latest ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${APP_NAME}:latest
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${APP_NAME}:latest
```

---
## 2) Create Aurora PostgreSQL (Serverless v2 recommended)
- Engine: **Aurora PostgreSQL** (choose a recent compatible version with `psycopg2` & PostgreSQL 12+ is fine).
- Publicly **unexposed** (private subnets).
- Create a **parameter group** that enforces SSL, or set `rds.force_ssl=1` (then the app will use `DB_SSLMODE=require`).
- Security groups:
  - **Aurora SG**: inbound from **ECS tasks SG** on port 5432 only.

Note your writer endpoint, e.g.: `lars-db.cuepfzx9q984.ap-northeast-2.rds.amazonaws.com`



---
## 3) Create ECS Cluster, TaskDefinition, Service
```bash
CLUSTER=lars-data-check-cluster
SERVICE=lars-data-check-svc
REGION=ap-northeast-2

aws ecs create-cluster --cluster-name ${CLUSTER} --region ${REGION} || true

# Replace placeholders in ecs-taskdef.template.json or do it manually in Console
# Then register the task definition:
aws ecs register-task-definition       --cli-input-json file://ecs-taskdef.template.json       --region ${REGION}

# Create security groups and ALB (you can use the console or CloudFormation).
# - ALB SG: inbound 80/443 from the world
# - ECS SG: inbound from ALB SG on 8080, outbound 0.0.0.0/0
# - ECS SG must be allowed to connect to Aurora SG on 5432

# Finally, create the ECS service (Console recommended):
# - Launch type: FARGATE
# - Desired tasks: 1+
# - VPC: select the same VPC as Aurora
# - Subnets: private subnets for tasks
# - Load balancer: ALB in public subnets, target port 8080
```

---
## 5) App Configuration (Environment)
Set these **environment variables** on the ECS Task (or via Task Definition):
- `DB_HOST` = your Aurora writer endpoint
- `DB_PORT` = `5432`
- `DB_NAME` = your database name
- `DB_USER` = DB username
- `DB_PWD`  = (from Secrets Manager *as a secret*)
- `DB_SSLMODE` = `require` (recommended)
- `SECRET_KEY` = a strong random string for Flask sessions

The app already reads these in `app.py` and builds a SQLAlchemy URI like:
`postgresql+psycopg2://DB_USER:DB_PWD@DB_HOST:DB_PORT/DB_NAME`

---
## 6) Health Check & Logs
- ALB target health check path can be `/` (the app renders a page).
- Logs go to CloudWatch: `/ecs/patient-app` (configured in taskdef).

---
## 7) Optional: Elastic Beanstalk (quickest path)
If you prefer EB:
- Create a new **Single Container Docker** environment.
- Upload this same Docker image or let EB build from `Dockerfile`.
- Put the env vars and Secrets Manager references in EB console.
- Ensure EB instances are in the same VPC/subnets with Aurora and SG rules allow 5432 to the Aurora SG.

---
## 8) Local test
```bash
cd patient_app_csv_testing
cp ../.env.sample .env  # fill values
python -m pip install -r ../requirements.txt
python - <<'PY'
from app import app
print('App loaded OK:', app)
PY

# Or run with gunicorn
gunicorn -c gunicorn.conf.py app:app
```

---
## Notes
- This app uses **SQLAlchemy + psycopg2**. Aurora PostgreSQL is wire-compatible, so no code changes are needed.
- If your Aurora enforces SSL, keep `DB_SSLMODE=require`. If not, you can omit it.
- Make sure your **ECS task IAM role** can read the Secrets Manager secret.
