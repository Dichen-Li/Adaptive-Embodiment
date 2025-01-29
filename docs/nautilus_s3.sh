"access_key": 9QQ35JQEJI0MGTEM0JSS
"secret_key": CiAil2Su7HmZIhhDBHdtB7mDxg70Ztr6jwMKhfcc
"endpoint": https://s3-haosu.nrp-nautilus.io

# cmd install
export AWS_ACCESS_KEY_ID=9QQ35JQEJI0MGTEM0JSS
export AWS_SECRET_ACCESS_KEY=CiAil2Su7HmZIhhDBHdtB7mDxg70Ztr6jwMKhfcc
export AWS_ENDPOINT_URL=https://s3-haosu.nrp-nautilus.io
# export AWS_ENDPOINT_URL=http://rook-ceph-rgw-haosu.rook-haosu	# If inside nautilus cluster:

apt update && apt install -y rclone
rclone config create ceph s3 env_auth true \
    access_key_id "$AWS_ACCESS_KEY_ID" \
    secret_access_key "$AWS_SECRET_ACCESS_KEY" \
    endpoint "$AWS_ENDPOINT_URL" \
    provider Ceph

#web GUI
rclone rcd --rc-web-gui

# creare and delete bucket
rclone mkdir s3_l2dai:new_bucket
rclone purge s3_l2dai:new_bucket

#list and check info
rclone lsd s3_l2dai: # check buckets
rclone ls s3_l2dai:panogen # check files
rclone size s3_l2dai:panogen # du -h

#upload
rclone -P copy ./Desktop/test.zip s3_l2dai:panogen # direct at panogen
rclone -P copy ./Desktop/test.zip s3_l2dai:panogen/test_dir # auto mkdir if no-exist
# 如果路径是文件夹，那么会把文件夹“下面”的所有文件/文件夹都copy到dest文件夹下。原文件夹不会出现。

#download
rclone -P copy s3_l2dai:panogen/test.zip ./Desktop # auto mkdir if no-exist
# 如果路径是文件夹，那么会把文件夹“下面”的所有文件/文件夹都copy到dest文件夹下。原文件夹不会出现。

#delete files
rclone delete s3_l2dai:panogen/test.zip # single file
rclone delete s3_l2dai:panogen/test_dir # dir

#rclone
apt-get install -y rclone
