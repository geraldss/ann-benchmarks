FROM ann-benchmarks

COPY --chmod=0755 ./annbench-linux-arm64 /home/gsann/gsann
RUN chmod +x /home/gsann/gsann

RUN mkfifo /tmp/datapipe
