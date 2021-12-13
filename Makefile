all: normal.cu streams.cu split_computation.cu
	nvcc -o normal normal.cu
	nvcc -o streams streams.cu
	nvcc -o split_computation split_computation.cu

normal: normal.cu
	nvcc -o normal normal.cu

streams: streams.cu
	nvcc -o streams streams.cu

split_computation: split_computation.cu
	nvcc -o split_computation split_computation.cu

clean:
	rm -fr *_output normal streams split_computation