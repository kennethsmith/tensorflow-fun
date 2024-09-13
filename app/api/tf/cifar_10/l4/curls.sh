
curl -X 'GET' \
  'http://127.0.0.1:8000/cifar_10/l4/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "Dog",
  "url": "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
}'
echo

curl -X 'GET' \
  'http://127.0.0.1:8000/cifar_10/l4/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "Frog",
  "url": "'"https://s3.${TFF_S3_REGION}.amazonaws.com/${TFF_S3_BUCKET}/cifar_10/images/00000-frog-train.png"'"
}'
echo

curl -X 'GET' \
  'http://127.0.0.1:8000/cifar_10/l4/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "Automobile",
  "url": "'"https://s3.${TFF_S3_REGION}.amazonaws.com/${TFF_S3_BUCKET}/cifar_10/images/10000-automobile-train.png"'"
}'
echo

curl -X 'GET' \
  'http://127.0.0.1:8000/cifar_10/l4/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "Ship",
  "url": "'"https://s3.${TFF_S3_REGION}.amazonaws.com/${TFF_S3_BUCKET}/cifar_10/images/20000-ship-train.png"'"
}'
echo