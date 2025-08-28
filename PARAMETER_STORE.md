# Using AWS Systems Manager Parameter Store (SecureString) for DB_PWD

## 1) Create the SecureString parameter
```bash
REGION=ap-northeast-2
aws ssm put-parameter       --name "/herings/lars-data-check/DB_PWD"       --type "SecureString"       --value "larsapienv!a"       --region ${REGION}
```

## 2) Task role permissions (attach this policy to your ECS task role)
Replace <ACCOUNT_ID>, <REGION> if needed.
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters"
      ],
      "Resource": "arn:aws:ssm:ap-northeast-2:757824880057:parameter/herings/lars-data-check/DB_PWD"
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "ssm.ap-northeast-2.amazonaws.com"
        }
      }
    }
  ]
}
```
Notes:
- By default, SecureString uses AWS‑managed key **alias/aws/ssm**. If you use a customer KMS key, restrict the `kms:Decrypt` resource to that key ARN.
- Your EC2 Container Service (ECS) **task role** needs these permissions, not the execution role.

## 3) Task definition
The `ecs-taskdef.template.json` is already updated to pull `DB_PWD` from Parameter Store via `"valueFrom": "arn:aws:ssm:..."`

## 4) Environment variables (no plain password)
Set on the task/container:
- `DB_HOST` (Aurora writer endpoint)
- `DB_PORT` = 5432
- `DB_NAME`
- `DB_USER`
- `DB_SSLMODE` = disable
- `SECRET_KEY` (Flask)

`DB_PWD` will be injected from SSM as a secret at runtime.
