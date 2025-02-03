FROM ann-benchmarks

COPY --chmod=0755 ./annbench-linux-arm64 /home/gsann/gsann
RUN chmod +x /home/gsann/gsann

RUN touch /home/gsann/data
RUN mkfifo /home/gsann/data.sig

RUN touch /home/gsann/queries
RUN mkfifo /home/gsann/queries.sig

RUN touch /home/gsann/results
RUN mkfifo /home/gsann/results.sig

RUN ulimit -c unlimited
RUN GOTRACEBACK=crash
