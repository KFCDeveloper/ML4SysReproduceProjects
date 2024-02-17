# scp -r ../build ubuntu@150.239.168.8:~/owk-actions

ssh ubuntu@150.239.168.8 'mkdir -p /home/ubuntu/owk-actions; rm -rf /home/ubuntu/owk-actions/build'
# rsync -rav -e ssh --include="*/" --include='*.zip' --exclude='*' \
# ../build ubuntu@150.239.168.8:/home/ubuntu/owk-actions
rsync -rav -e ssh --include="*/" ../build ubuntu@150.239.168.8:/home/ubuntu/owk-actions