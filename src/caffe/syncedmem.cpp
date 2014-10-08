#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


	SyncedMemory::~SyncedMemory() {
		if (cpu_data_) {
			CaffeFreeHost(cpu_data_);
		}
#ifndef CPU_ONLY
		if (gpu_data_) {
			CUDA_CHECK(cudaFree(gpu_data_));
		}
#endif // CPU_ONLY
	}

	inline void SyncedMemory::to_cpu() {
		switch (head_) {
		case UNINITIALIZED:
			cpu_resize();
			head_ = HEAD_AT_CPU;
			break;
		case HEAD_AT_GPU:
#ifndef CPU_ONLY 
			switch(Caffe::gpu_mode())
			{
			case Caffe::GPU_AVAILABLE:
				gpu_resize();
				cpu_resize();
				caffe_gpu_memcpy(size_, gpu_data_, cpu_data_);
				//CUDA_CHECK(cudaMemcpy(cpu_data_, gpu_data_, size_, cudaMemcpyDeviceToHost));
				head_ = SYNCED;
				break;

			case Caffe::GPU_FORBID:
				cpu_resize();
				caffe_gpu_memcpy(std::min(size_, gpu_capacity_), gpu_data_, cpu_data_);
				//CUDA_CHECK(cudaMemcpy(cpu_data_, gpu_data_, std::min(size_, gpu_capacity_), cudaMemcpyDeviceToHost));
				head_ = HEAD_AT_CPU;
				break;

			default:
				LOG(FATAL) << "Unknown caffe gpu_mode.";
			}
#else
			NO_GPU
#endif
			break;
		case HEAD_AT_CPU:
			cpu_resize();
			break;
		case SYNCED:
			if (cpu_resize()) {
				head_ = HEAD_AT_CPU;
			}
			break;
		}
	}

	inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
		if (Caffe::gpu_mode() ==  Caffe::GPU_FORBID)
			LOG(ERROR) << "Caffe::FORBID can't get gpu data.";

		switch (head_) {
		case UNINITIALIZED:
			gpu_resize();
			head_ = HEAD_AT_GPU;
			break;
		case HEAD_AT_CPU:
			cpu_resize();
			gpu_resize();
			caffe_gpu_memcpy(size_, cpu_data_, gpu_data_);
			//CUDA_CHECK(cudaMemcpy(gpu_data_, cpu_data_, size_, cudaMemcpyHostToDevice));
			head_ = SYNCED;
			break;
		case HEAD_AT_GPU:
			gpu_resize();
			break;
		case SYNCED:
			if (gpu_resize()) {
				head_ = HEAD_AT_GPU;
			}
			break;
		}
#else
		NO_GPU;
#endif
	}


	const void* SyncedMemory::cpu_data() {
		to_cpu();
		return static_cast<const void*>(cpu_data_);
	}

	const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
		to_gpu();
		return static_cast<const void*>(gpu_data_);
#else
		NO_GPU;
		return NULL;
#endif

	}

	void* SyncedMemory::mutable_cpu_data() {
		to_cpu();
		head_ = HEAD_AT_CPU;
		return cpu_data_;
	}

	void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
		to_gpu();
		head_ = HEAD_AT_GPU;
		return gpu_data_;
#else
		NO_GPU;
		return NULL;
#endif
	}

	// If host (CPU) memory is uninitialized or cpu_capacity_ < size_, allocate the
	// appropriate amount of additional host memory. Otherwise, do nothing.
	// Returns the number of extra bytes allocated.
	size_t SyncedMemory::cpu_resize() {
		if (!cpu_data_) {
			CaffeMallocHost(&cpu_data_, size_);
		} else if (size_ > cpu_capacity_) {
			CaffeReallocHost(&cpu_data_, size_);
		} else {
			return 0;
		}
		size_t num_new_bytes = size_ - cpu_capacity_;
		// Zero-fill memory starting from offset cpu_capacity_ (i.e., don't overwrite
		// current data).
		memset(static_cast<uint8_t*>(cpu_data_) + cpu_capacity_, 0, num_new_bytes);
		cpu_capacity_ = size_;
		return num_new_bytes;
	}

	// If GPU device memory is uninitialized or gpu_capacity_ < size_, allocate the
	// appropriate amount of additional GPU memory. Otherwise, do nothing.
	// Returns the number of extra bytes allocated.
	size_t SyncedMemory::gpu_resize() {
		if (!gpu_data_) {
			CUDA_CHECK(cudaMalloc(&gpu_data_, size_));
		} else if (size_ > gpu_capacity_) {
			void* new_gpu_data;
			CUDA_CHECK(cudaMalloc(&new_gpu_data, size_));
			caffe_gpu_memcpy(gpu_capacity_, gpu_data_, new_gpu_data);
			//CUDA_CHECK(cudaMemcpy(new_gpu_data, gpu_data_, gpu_capacity_,
			//	cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaFree(gpu_data_));
			gpu_data_ = new_gpu_data;
		} else {
			return 0;
		}
		size_t num_new_bytes = size_ - gpu_capacity_;
		// Zero-fill memory starting from offset gpu_capacity_ (i.e., don't overwrite
		// current data).
		caffe_gpu_memset(num_new_bytes, 0, static_cast<uint8_t*>(gpu_data_) + gpu_capacity_);
		//CUDA_CHECK(cudaMemset(
		//	static_cast<uint8_t*>(gpu_data_) + gpu_capacity_, 0, num_new_bytes));
		gpu_capacity_ = size_;
		return num_new_bytes;
	}


}  // namespace caffe

