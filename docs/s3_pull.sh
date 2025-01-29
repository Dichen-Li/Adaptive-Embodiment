cd /embodiment-scaling-law/exts/berkeley_humanoid/berkeley_humanoid/assets/Robots && \
rclone -P copy ceph:cross-em-genbot1k-v1/GenBot1K-v1.zip ./ && unzip GenBot1K-v1.zip && \
cd ../../../../../ && \
mkdir compressed_logs && cd compressed_logs && \
rclone -P copy ceph:cross-em-genbot1k-v1/bai/logs ./bai && \
rclone -P copy ceph:cross-em-genbot1k-v1/liudai/logs ./liudai  && \
rclone -P copy ceph:cross-em-genbot1k-v1/dichen/logs ./dichen && \
rclone -P copy ceph:cross-em-genbot1k-v1/tmu/logs ./tmu && \