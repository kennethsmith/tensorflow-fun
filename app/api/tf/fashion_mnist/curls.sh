
curl -X 'GET' \
  'http://127.0.0.1:8000/fashion_mnist/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "dog",
  "url": "'"https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"'"
}'
echo

curl -X 'GET' \
  'http://127.0.0.1:8000/fashion_mnist/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "Ankle Boot",
  "url": "'"https://s3.${TFF_S3_REGION}.amazonaws.com/${TFF_S3_BUCKET}/fashion_mnist/images/00000-Ankle+boot-train.jpg"'"
}'
echo

curl -X 'GET' \
  'http://127.0.0.1:8000/fashion_mnist/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "Bag",
  "url": "'"https://s3.${TFF_S3_REGION}.amazonaws.com/${TFF_S3_BUCKET}/fashion_mnist/images/10000-Bag-train.jpg"'"
}'
echo

curl -X 'GET' \
  'http://127.0.0.1:8000/fashion_mnist/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "Sneaker",
  "url": "'"https://s3.${TFF_S3_REGION}.amazonaws.com/${TFF_S3_BUCKET}/fashion_mnist/images/20000-Sneaker-train.jpg"'"
}'
echo