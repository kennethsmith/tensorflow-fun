tensorflow_fun="<Set your values here. Where is this repo on your system.>"
python_path="<Set your values. What is the path to your Python installation.>"
tff_s3_bucket="<Set your values. What is your S3 bucket name?>"
tff_s3_profile="<Set your values. What AWS credentials profile are you using?>"
tff_s3_region="<Set your values. What region is your bucket in?>"

timestamp=$(date +%s)

cp ~/.zprofile ~/.zprofile.$timestamp

echo '\n' >> ~/.zprofile
echo "# Start of tensorflow-fun changes" >> ~/.zprofile
echo '\n' >> ~/.zprofile
echo 'TFF_S3_BUCKET="'${tff_s3_bucket}'"' >> ~/.zprofile
echo 'TFF_S3_PROFILE="'${tff_s3_profile}'"' >> ~/.zprofile
echo 'TFF_S3_REGION="'${tff_s3_region}'"' >> ~/.zprofile
echo 'TFF_VENV="'${tensorflow_fun}'/venv/bin"' >> ~/.zprofile
echo 'PYTHON_PATH="'${python_path}'"' >> ~/.zprofile
echo 'PATH="${TFF_VENV}:${PATH}"' >> ~/.zprofile
echo '\n' >> ~/.zprofile
echo 'export TFF_S3_BUCKET' >> ~/.zprofile
echo 'export TFF_S3_PROFILE' >> ~/.zprofile
echo 'export TFF_S3_REGION' >> ~/.zprofile
echo 'export TFF_VENV' >> ~/.zprofile
echo 'export PYTHON_PATH' >> ~/.zprofile
echo 'export PATH' >> ~/.zprofile
echo '\n' >> ~/.zprofile
echo "# Ene of tensorflow-fun changes" >> ~/.zprofile
