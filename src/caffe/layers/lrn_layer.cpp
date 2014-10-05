#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void LRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			size_ = this->layer_param_.lrn_param().local_size();
			CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
			pre_pad_ = (size_ - 1) / 2;
			alpha_ = this->layer_param_.lrn_param().alpha();
			beta_ = this->layer_param_.lrn_param().beta();
	}

	template <typename Dtype>
	void LRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			num_ = bottom[0]->num();
			channels_ = bottom[0]->channels();
			height_ = bottom[0]->height();
			width_ = bottom[0]->width();
			(*top)[0]->Reshape(num_, channels_, height_, width_);
			scale_.Reshape(num_, channels_, height_, width_);
	}

	template <typename Dtype>
	void LRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			switch (this->layer_param_.lrn_param().norm_region()) {
			case LRNParameter_NormRegion_ACROSS_CHANNELS:
				CrossChannelForward_cpu(bottom, top);
				break;
			case LRNParameter_NormRegion_WITHIN_CHANNEL:
				WithinChannelForward_cpu(bottom, top);
				break;
			default:
				LOG(FATAL) << "Unknown normalization region.";
			}
	}

	template <typename Dtype>
	void LRNLayer<Dtype>::CrossChannelForward_cpu(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* top_data = (*top)[0]->mutable_cpu_data();
			Dtype* scale_data = scale_.mutable_cpu_data();
			// start with the constant value
			for (int i = 0; i < scale_.count(); ++i) {
				scale_data[i] = 1.;
			}
			Blob<Dtype> padded_square(1, channels_ + size_ - 1, height_, width_);
			Dtype* padded_square_data = padded_square.mutable_cpu_data();
			caffe_set(padded_square.count(), Dtype(0), padded_square_data);
			Dtype alpha_over_size = alpha_ / size_;
			// go through the images
			for (int n = 0; n < num_; ++n) {
				// compute the padded square
				caffe_sqr(channels_ * height_ * width_,
					bottom_data + bottom[0]->offset(n),
					padded_square_data + padded_square.offset(0, pre_pad_));
				// Create the first channel scale
				for (int c = 0; c < size_; ++c) {
					caffe_axpy<Dtype>(height_ * width_, alpha_over_size,
						padded_square_data + padded_square.offset(0, c),
						scale_data + scale_.offset(n, 0));
				}
				for (int c = 1; c < channels_; ++c) {
					// copy previous scale
					caffe_copy<Dtype>(height_ * width_,
						scale_data + scale_.offset(n, c - 1),
						scale_data + scale_.offset(n, c));
					// add head
					caffe_axpy<Dtype>(height_ * width_, alpha_over_size,
						padded_square_data + padded_square.offset(0, c + size_ - 1),
						scale_data + scale_.offset(n, c));
					// subtract tail
					caffe_axpy<Dtype>(height_ * width_, -alpha_over_size,
						padded_square_data + padded_square.offset(0, c - 1),
						scale_data + scale_.offset(n, c));
				}
			}

			// In the end, compute output
			caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, top_data);
			caffe_mul<Dtype>(scale_.count(), top_data, bottom_data, top_data);
	}

	template <typename Dtype>
	void LRNLayer<Dtype>::WithinChannelForward_cpu(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* top_data = (*top)[0]->mutable_cpu_data();
			Dtype* scale_data = scale_.mutable_cpu_data();
			// start with the constant value
			for (int i = 0; i < scale_.count(); ++i) {
				scale_data[i] = 1.;
			}
			Blob<Dtype> square(1, channels_, height_, width_);
			Dtype* square_data = square.mutable_cpu_data();
			Dtype alpha_over_size = alpha_ / (size_ * size_);

			int half_window = (size_ - 1) / 2;
			int pixels = width_ * height_;

			// go through the images
			for (int n = 0; n < num_; ++n) {
				// compute the square
				caffe_sqr(square.count(),
					bottom_data + bottom[0]->offset(n),
					square_data);

				for (int h = 0; h < height_; h++)
				{
					int lh = max(h - half_window, 0);
					int hh = min(h - half_window + size_, height_);
					for (int w = 0; w < width_; w++)
					{
						int lw = max(w - half_window, 0);
						int hw = min(w - half_window + size_, width_);
						for (int dh = lh; dh < hh; dh++)
						{
							for (int dw = lw; dw < hw; dw++)
							{
								caffe_axpy_step(channels_, alpha_over_size,
									square_data + square.offset(0, 0, dh, dw), pixels,
									scale_data + scale_.offset(n, 0, h, w), pixels);
							}
						}

					}
				}
			}

			// In the end, compute output
			caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, top_data);
			caffe_mul<Dtype>(scale_.count(), top_data, bottom_data, top_data);
	}

	template <typename Dtype>
	void LRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
			switch (this->layer_param_.lrn_param().norm_region()) {
			case LRNParameter_NormRegion_ACROSS_CHANNELS:
				CrossChannelBackward_cpu(top, propagate_down, bottom);
				break;
			case LRNParameter_NormRegion_WITHIN_CHANNEL:
				WithinChannelBackward_cpu(top, propagate_down, bottom);
				break;
			default:
				LOG(FATAL) << "Unknown normalization region.";
			}
	}

	template <typename Dtype>
	void LRNLayer<Dtype>::CrossChannelBackward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		vector<Blob<Dtype>*>* bottom) {
			const Dtype* top_diff = top[0]->cpu_diff();
			const Dtype* top_data = top[0]->cpu_data();
			const Dtype* bottom_data = (*bottom)[0]->cpu_data();
			const Dtype* scale_data = scale_.cpu_data();
			Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
			Blob<Dtype> padded_ratio(1, channels_ + size_ - 1, height_, width_);
			Blob<Dtype> accum_ratio(1, 1, height_, width_);
			Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
			Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
			// We hack a little bit by using the diff() to store an additional result
			Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
			caffe_set(padded_ratio.count(), Dtype(0), padded_ratio_data);
			Dtype cache_ratio_value = 2. * alpha_ * beta_ / size_;

			caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, bottom_diff);
			caffe_mul<Dtype>(scale_.count(), top_diff, bottom_diff, bottom_diff);

			// go through individual data
			int inverse_pre_pad = size_ - (size_ + 1) / 2;
			for (int n = 0; n < num_; ++n) {
				int block_offset = scale_.offset(n);
				// first, compute diff_i * y_i / s_i
				caffe_mul<Dtype>(channels_ * height_ * width_,
					top_diff + block_offset, top_data + block_offset,
					padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
				caffe_div<Dtype>(channels_ * height_ * width_,
					padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad),
					scale_data + block_offset,
					padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
				// Now, compute the accumulated ratios and the bottom diff
				caffe_set(accum_ratio.count(), Dtype(0), accum_ratio_data);
				for (int c = 0; c < size_ - 1; ++c) {
					caffe_axpy<Dtype>(height_ * width_, 1.,
						padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
				}
				for (int c = 0; c < channels_; ++c) {
					caffe_axpy<Dtype>(height_ * width_, 1.,
						padded_ratio_data + padded_ratio.offset(0, c + size_ - 1),
						accum_ratio_data);
					// compute bottom diff
					caffe_mul<Dtype>(height_ * width_,
						bottom_data + top[0]->offset(n, c),
						accum_ratio_data, accum_ratio_times_bottom);
					caffe_axpy<Dtype>(height_ * width_, -cache_ratio_value,
						accum_ratio_times_bottom, bottom_diff + top[0]->offset(n, c));
					caffe_axpy<Dtype>(height_ * width_, -1.,
						padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
				}
			}
	}

	template <typename Dtype>
	void LRNLayer<Dtype>::WithinChannelBackward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		vector<Blob<Dtype>*>* bottom) {
			if (propagate_down[0]) {
				const Dtype* top_diff = top[0]->cpu_diff();
				const Dtype* top_data = top[0]->cpu_data();
				const Dtype* bottom_data = (*bottom)[0]->cpu_data();
				const Dtype* scale_data = scale_.cpu_data();
				Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
				Blob<Dtype> ratio(1, channels_, height_, width_);
				Blob<Dtype> accum_ratio(1, channels_, 1, 1);
				Dtype* ratio_data = ratio.mutable_cpu_data();
				Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
				// We hack a little bit by using the diff() to store an additional result
				Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
				Dtype cache_ratio_value = 2. * alpha_ * beta_ / (size_ * size_);

				caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, bottom_diff);
				caffe_mul<Dtype>(scale_.count(), top_diff, bottom_diff, bottom_diff);

				int pixels = width_ * height_;
				int half_window = (size_ - 1) / 2;

				// go through individual data
				for (int n = 0; n < num_; ++n) 
				{
					int block_offset = scale_.offset(n);
					// first, compute diff_i * y_i / s_i
					caffe_mul<Dtype>(channels_ * height_ * width_,
						top_diff + block_offset, top_data + block_offset,
						ratio_data);
					caffe_div<Dtype>(channels_ * height_ * width_,
						ratio_data,
						scale_data + block_offset,
						ratio_data);
					// Now, compute the accumulated ratios and the bottom diff

					for (int h = 0; h < height_; h++)
					{
						int lh = max(h - half_window, 0);
						int hh = min(h - half_window + size_, height_);
						for (int w = 0; w < width_; w++)
						{
							int lw = max(w - half_window, 0);
							int hw = min(w - half_window + size_, width_);

							memset(accum_ratio_data, 0, sizeof(Dtype) * accum_ratio.count());

							for (int dh = lh; dh < hh; dh++)
							{
								for (int dw = lw; dw < hw; dw++)
								{
									caffe_axpy_step<Dtype>(channels_, 1., 
										ratio_data + ratio.offset(0, 0, dh, dw), pixels,
										accum_ratio_data, 1);
								}
							}

							// compute bottom diff

							const Dtype *xi = bottom_data + top[0]->offset(n, 0, h, w);
							for (int i=0; i<channels_; i++)
							{
								accum_ratio_times_bottom[i] = *xi * accum_ratio_data[i];
								xi += pixels;
							}
							caffe_axpy_step(channels_, -cache_ratio_value,
								accum_ratio_times_bottom, 1, 
								bottom_diff + top[0]->offset(n, 0, h, w), pixels);
						}
					}
				}
			}
	}

#ifdef CPU_ONLY
	STUB_GPU(LRNLayer);
	STUB_GPU_FORWARD(LRNLayer, CrossChannelForward);
	STUB_GPU_BACKWARD(LRNLayer, CrossChannelBackward);
	STUB_GPU_FORWARD(LRNLayer, WithinChannelForward);
	STUB_GPU_BACKWARD(LRNLayer, WithinChannelBackward);
#endif

	INSTANTIATE_CLASS(LRNLayer);


}  // namespace caffe
