aws s3 sync \
   ./ \
   s3://${TFF_S3_BUCKET}/fashion_mnist/ \
   --exclude ".*" \
   --exclude "*.py" \
   --exclude "*.sh" \
   --exclude "*.html" \
   --acl public-read \
   --profile ${TFF_S3_PROFILE}
   